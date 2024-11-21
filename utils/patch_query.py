import time
import trimesh
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pyvista as pv
import sys

from pyvistaqt import QtInteractor
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from scipy.spatial import cKDTree as KDTree
import pandas as pd
import random


class PatchQuery:
    def __init__(self, patches, query_threshold=50):
        self.patches = patches
        self.query_threshold = query_threshold
        self.patchFeatureMatrix = self.get_hausdorff_matrix()

        self.patchQR = self.clustering_hks()  # (x_embedded, y_pred)

    def align_and_normalize(self, mesh: pv.PolyData):
        mesh = trimesh.Trimesh(vertices=mesh.points, faces=np.array(mesh.faces.reshape(-1, 4)[:, 1:]))
        centroid = mesh.bounds.mean(axis=0)
        translation_matrix = trimesh.transformations.translation_matrix(-centroid)
        mesh.apply_transform(translation_matrix)

        scale_factor = 1.0 / max(mesh.extents)  # Scale by the maximum extent
        scale_matrix = trimesh.transformations.scale_matrix(scale_factor)
        mesh.apply_transform(scale_matrix)

        return pv.wrap(mesh)

    def get_hausdorff_distance(self, mesh1, mesh2):
        distance1 = directed_hausdorff(mesh1.points, mesh2.points)[0]
        distance2 = directed_hausdorff(mesh2.points, mesh1.points)[0]
        return max(distance1, distance2)

    def align_meshes_icp(self, mesh1, mesh2):
        # 获取mesh1和mesh2的点云
        points1 = mesh1.points
        points2 = mesh2.points

        mesh1 = trimesh.Trimesh(vertices=points1, faces=np.array(mesh1.faces.reshape(-1, 4)[:, 1:]))
        mesh2 = trimesh.Trimesh(vertices=points2, faces=np.array(mesh2.faces.reshape(-1, 4)[:, 1:]))

        # 使用ICP算法进行配准
        reg, _ = trimesh.registration.mesh_other(mesh1, mesh2, icp_first=15, samples=500)

        # 获取配准后的mesh

        aligned_mesh2 = mesh2.apply_transform(reg)

        return aligned_mesh2

    def get_hausdorff_matrix(self):
        num_meshs = len(self.patches)
        hausdorff_matrix = np.zeros((num_meshs, num_meshs))
        mesh_list = [self.align_and_normalize(pv.wrap(patch)) for patch in self.patches]

        patch_list = []
        patch_list.append(mesh_list[0])
        for mesh in mesh_list[1:]:
            mesh = self.align_meshes_icp(patch_list[0], mesh)
            patch_list.append(mesh)

        for i in range(num_meshs):
            for j in range(i + 1, num_meshs):
                mesh1 = patch_list[i]
                mesh2 = patch_list[j]
                mesh1, mesh2 = pv.wrap(mesh1), pv.wrap(mesh2)
                distance = self.get_hausdorff_distance(mesh1, mesh2)
                hausdorff_matrix[i, j] = distance
                hausdorff_matrix[j, i] = distance
            print(f"Finish {i}th row")

        return hausdorff_matrix

    def clustering_hks(self, plot=False):
        x_embedded = TSNE(perplexity=3, init="pca").fit_transform(self.patchFeatureMatrix)
        ward = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=50,
                                       linkage="ward").fit(x_embedded)
        y_pred = ward.labels_
        if plot:
            plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y_pred, cmap='Spectral')
            plt.show()
        return x_embedded, y_pred
