from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from utils.patch_partition import Patcher
# from patch_query import PatchQuery
import numpy as np
import trimesh
import pathlib


class MulQR:
    def __init__(self, patch, patch_level_hks, meshDict, query_threshold=50, size_threshold=50):
        """
        cluster the selected patch with patches from other meshes.
        :param meshes: each element is a dictionary, which contains the
        mesh_pth, tsne_pth, hks_pth
        """
        self.patch = patch
        self.patch_level_hks = patch_level_hks
        self.meshdict = meshDict
        self.query_threshold = query_threshold
        self.size_threshold = size_threshold

    def get_patchInfo(self):
        patcher = Patcher(self.meshdict["mesh_pth"], self.meshdict["tsne_pth"], self.meshdict["hks_pth"],
                          self.size_threshold)
        patcher.get_surfpatch()
        patchQR = patcher.get_patchInfo()
        self.meshdict["QR"] = patchQR

    def aggregrate_mesh(self):
        """
        aggregrate the selected patch with another mesh.
        :return: dictionary (patch list, hks matrix)
        """
        patch_list = []
        hks_matrix = []

        patch_list.append(self.patch)
        hks_matrix.append(self.patch_level_hks)

        patchDict, patch_hks, _ = self.meshdict["QR"]  # mesh info
        patch_num = len(patchDict)
        for i in range(patch_num):
            patch_list.append(patchDict[i])
            hks_matrix.append(patch_hks[i])

        hks_matrix = np.array(hks_matrix)
        return patch_list, hks_matrix

    def run(self):
        """
        Kernel function of the class
        :return: query result list [patch1, patch2, ...]
        """
        self.get_patchInfo()
        patch_list, hks_matrix = self.aggregrate_mesh()
        x_embedded, y_pred = self.clustering(hks_matrix)

        # query result is saved in a dictionary
        query_list = list()
        query_num = np.unique(y_pred).shape[0]
        for i in range(query_num):
            idx = np.where(y_pred == i)[0]
            if 0 in idx:
                query_list = [patch_list[j] for j in idx if j != 0]
                break

        return query_list

    def clustering(self, hks_matrix):
        x_embedded = TSNE(perplexity=3).fit_transform(hks_matrix)
        ward = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=self.query_threshold,
                                       linkage="ward").fit(x_embedded)
        y_pred = ward.labels_
        return x_embedded, y_pred
