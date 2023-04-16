from occwl.jupyter_viewer import JupyterViewer
from multiprocessing import Pool, set_start_method
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pandas as pd
import json

 
class Plotter:
    def __init__(self, name_map=None):
        if name_map is not None:
            self.name_map = name_map
            self.index = {k: i for i,k in enumerate(self.name_map)}
        else:
            self.name_map = pd.read_csv("data/MFCAD++_dataset/feature_labels.txt", skiprows=6, header=None, sep=" - ").set_index(0)[1].to_dict()

    def plot_predict(self, data, labels):
        """Plot gt - plots additional shape with ground truth labels"""
        v = JupyterViewer()
        norm = Normalize(min(labels), max(labels))
        norm_values_for_faces = norm([self.index[x] for x in labels])

        color_mapper = get_cmap('rainbow')
        face_colors = color_mapper(norm_values_for_faces)[:, :3]

        v.display_face_colormap(data['shape'], values_for_faces=[self.index[x] for x in labels])
        fig, ax = plt.subplots()
        fig.set_size_inches(.5, .5)
        # Add labels to the legend without plotting data

        processed = []
        for label,color in zip(labels, face_colors):
            if self.name_map[label] not in processed:
                processed.append(self.name_map[label])
                ax.scatter([], [], label=self.name_map[label], color=color)

        # Add the legend
        ax.legend(loc='upper left', facecolor='white', framealpha=1, fancybox=True, edgecolor='gray', fontsize='large')

        # Turn off axis labels and ticks
        ax.axis('off')

        # Show the legend
        plt.show()
        v.show()
    
    @staticmethod
    def plot_truth(data):
        """Plot gt - plots additional shape with ground truth labels"""
        v = JupyterViewer()
        v.display_face_colormap(data['shape'], values_for_faces=data['graph'].ndata['y'])
        v.show()
        
