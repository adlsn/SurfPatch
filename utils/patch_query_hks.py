import time
import trimesh
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pyvista as pv
import sys

from pyvistaqt import QtInteractor
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE, Isomap, MDS

from scipy.spatial import cKDTree as KDTree
import pandas as pd
import random

from umap import UMAP


def generate_random_hex_color():
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color


class PatchQuery:
    def __init__(self, mesh, patchDict, patchFeature: "matrix", query_threshold=50):
        self.mesh = mesh
        self.patchDict = patchDict
        self.patchFeature = patchFeature
        self.query_threshold = query_threshold

        self.patchQR = self.clustering_hks()  # (x_embedded, y_pred)

    def clustering_hks(self, plot=False):

        x_embedded = TSNE(perplexity=3, n_iter=2000, init="pca").fit_transform(self.patchFeature)
        ward = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=self.query_threshold,
                                       linkage="ward").fit(x_embedded)
        y_pred = ward.labels_
        if plot:
            plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y_pred, cmap='Spectral')
            plt.show()
        return x_embedded, y_pred

    @staticmethod
    def visualize_patch(patch: trimesh.Trimesh):
        mesh = pv.wrap(patch)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='red', show_edges=True)
        plotter.show()

    @staticmethod
    def visualize_multi_patch(patches: list[trimesh.Trimesh]):
        plotter = pv.Plotter()
        for patch in patches:
            mesh = pv.wrap(patch)
            plotter.add_mesh(mesh, color=generate_random_hex_color(), show_edges=False, opacity=0.7)

        plotter.show()

    @staticmethod
    def visualize_video(patches):
        for patch in patches:
            mesh = pv.wrap(patch)
            pvwidget.add_mesh(mesh, color=generate_random_hex_color(), show_edges=False, opacity=0.7)
        pvwidget.show()

