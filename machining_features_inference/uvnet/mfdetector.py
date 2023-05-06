from logging import Logger

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from tqdm import tqdm

from machining_features_inference.evaluation.jaccard import get_mf_jaccard
from machining_features_inference.settings import UVNetPipelineSettings
from machining_features_inference.uvnet.models import UVNetSegmenter


class MFDetector:
    """
    module to train/test the segmenter (per-face classifier).
    """

    def __init__(self, settings: UVNetPipelineSettings, logger: Logger):
        """
        Args:
            num_classes (int): Number of per-face classes in the dataset
        """
        self._logger = logger
        self._settings = settings
        self.device = torch.device(settings.device)
        self.model = UVNetSegmenter(
            settings.num_classes,
            crv_in_channels=settings.crv_in_channels,
            dropout=settings.dropout,
            srf_emb_dim=settings.srf_emb_dim,
            crv_emb_dim=settings.crv_emb_dim,
        )
        self.model = self.model.to(device=self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=settings.lr,
            weight_decay=settings.lr,
        )

        self.feature_weights = torch.ones(
            settings.num_classes, device=self.device, dtype=torch.float
        )
        for ifeat in settings.skip_labels:
            self.feature_weights[ifeat] = 0.0
        self.add_bias = torch.zeros(
            settings.num_classes, device=self.device, dtype=torch.float
        )
        for ifeat in settings.skip_labels:
            self.add_bias[ifeat] = float("-inf")

        self.val_iou = torchmetrics.IoU(
            num_classes=settings.num_classes, compute_on_step=False
        )

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)

        loss = F.cross_entropy(logits, labels, weight=self.feature_weights)
        return loss

    @torch.no_grad()
    def calc_step(self, batch, valid_preds):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)

        softmax = F.softmax(logits + self.add_bias, dim=-1)

        preds = softmax.argmax(dim=1).cpu().numpy().tolist()
        valid_preds[batch["filename"]] = preds

    def train(self, data_loader):
        self.model.train()
        for batch_idx, batch in tqdm(
            enumerate(data_loader), desc="Training", total=len(data_loader)
        ):
            self.optimizer.zero_grad()

            loss = self.training_step(batch, batch_idx)

            loss.backward()
            self.optimizer.step()
        self._logger.info(f"loss: {loss.item()}")

    @torch.no_grad()
    def valid(self, data_loader):
        self.model.eval()

        for batch_idx, batch in tqdm(
            enumerate(data_loader), desc="test", total=len(data_loader)
        ):
            logits, _, labels = self.test_step(batch)
            loss = F.cross_entropy(logits, labels, weight=self.feature_weights)
        self._logger.info(f"loss: {loss.item()},  iou: {self.val_iou.compute()}")

    @torch.no_grad()
    def test_jaccared(self, dataset):
        self.model.eval()
        valid_preds = dict()
        for batch_idx, batch in tqdm(
            enumerate(dataset), desc="Validation", total=len(dataset)
        ):
            self.calc_step(batch, valid_preds)

        jaccards = []
        for sample in dataset:
            flnm = sample["filename"]
            if flnm in valid_preds:
                jaccards.append(get_mf_jaccard(sample=sample, labels=valid_preds[flnm]))
        self._logger.info("Jaccard:", np.mean(jaccards))
        return np.mean(jaccards)

    def test_step(self, batch):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        softmax = F.softmax(logits + self.add_bias, dim=-1)
        preds = softmax.argmax(dim=1)

        self.val_iou(preds.cpu(), labels.cpu())
        return logits, preds, labels

    @torch.no_grad()
    def test_matr(self, dloader):
        self.model.eval()
        lbs_list = []
        preds_list = []
        for batch in dloader:
            _, preds, labels = self.test_step(batch)
            lbs_list.append(labels.cpu())
            preds_list.append(preds.cpu())

        return self.confmat(torch.cat(lbs_list), torch.cat(preds_list))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self._logger.info(f"Model saved to path: {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self._logger.info(f"Model loaded from path: {path}")

    def get_labels(self, dataset):
        rez_list = []
        with torch.no_grad():
            for data in dataset:
                rz = dict()
                rz["part"] = data["filename"]
                inputs = data["graph"].to(self.device)
                inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
                inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)

                logits = self.model(inputs).to(device=torch.device("cpu"))
                softmax = F.softmax(logits, dim=-1)

                rz["labels"] = softmax.argmax(dim=1).numpy().tolist()
                rez_list.append(rz)
        return rez_list
