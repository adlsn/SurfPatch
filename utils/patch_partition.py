import random

import trimesh
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
import pathlib
from pyvistaqt import QtInteractor
from pyvistaqt import MultiPlotter
import time


class Patcher:
    def __init__(self, mesh_path, tsne_path, hks_path, distance_threshold=50):
        if not mesh_path.endswith('.stl') or not tsne_path.endswith('.npy'): # TODO: origin is .stl
            raise ValueError("The mesh path or feature path is not valid!!!")

        self.mesh = pv.read(mesh_path)
        self.features = np.load(tsne_path)  # TODO: compare tsne with other DR methods
        self.hks = np.load(hks_path)

        # operate on the mesh

        # ================== hierarchy cluster ==================
        self.connectivity_matrix = self.get_mesh_connectivity()
        self.x_embedded, self.clusterLabel = self.hierarchy_cluster(self.features, plot=False,
                                                                    distance_threshold=distance_threshold)



    def hierarchy_cluster(self, feature_matrix, distance_threshold, *, plot=False):
        """
        Cluster the features using hierarchy cluster. This method is based on the
        following link: https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py
        :param feature_matrix: in this case, it is the heat kernel signature
        :param plot: plot the result by 2D vision
        :return: cluster label matrix, n x 1, n is the number of vertices in the mesh
        """
        x_embedded = feature_matrix
        # ward = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=distance_threshold,
        #                                linkage="ward", connectivity=self.connectivity_matrix).fit(x_embedded)
        ward = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=distance_threshold,
                                       linkage="ward").fit(x_embedded)
        y_pred = ward.labels_

        # =======comparison=======
        # ms = MeanShift(bandwidth=2)
        # ms.fit(x_embedded)
        # y_pred = ms.labels_

        # if plot:
        #     plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y_pred, cmap='Spectral')
        #     plt.show()
        return x_embedded, y_pred

    def get_mesh_connectivity(self):
        """
        get the connectivity of the mesh
        :return: ndarray, n x n, n is the number of vertices in the mesh
        """
        t_mesh = trimesh.Trimesh(self.mesh.points, self.mesh.faces.reshape(-1, 4)[:, 1:])
        connectivity_matrix = t_mesh.edges_sparse
        return connectivity_matrix

    @staticmethod
    def visualize_patch(mesh, clusterLabel):
        """
        visualize the patch result for any partition method.
        :param mesh: pyvista.PolyData
        :param clusterLabel: ndarray, n x 1, n is the number of faces in the mesh
        :return: None
        """
        patch_num = np.unique(clusterLabel)
        for i in patch_num:
            plotter = pv.Plotter()
            plotter.add_mesh(mesh, style='wireframe', color='white')
            vertexID = np.where(clusterLabel == i)[0]
            pc = mesh.points[vertexID]
            plotter.add_points(pc, color='red', point_size=15)
            plotter.show()
            print(plotter.camera_position)

    def visualize_patch_video(self, mesh, clusterLabel):
        """
        visualize the patch result for any partition method.
        :param mesh: pyvista.PolyData
        :param clusterLabel: ndarray, n x 1, n is the number of faces in the mesh
        :return: None
        """
        # pvwidget = QtInteractor() # this is for single window
        pvwidget = MultiPlotter(nrows=1, ncols=2)
        pvwidget.window_size = [800, 800]
        patch_num = np.unique(clusterLabel)
        for i in patch_num:
            pvwidget[0, 0].clear_actors()
            pvwidget[0, 1].clear_actors()
            # two swirls
            # pvwidget[0, 0].camera_position = [(46.92120074398903, 117.4536088140038, 21.86087709014464),
            #                                   (20.114616870880127, 17.838438272476196, 27.890840008854866),
            #                                   (0.9657187024954271, -0.2595264844008965, 0.005778541776276807)]
            # pvwidget[0, 1].camera_position = [(46.92120074398903, 117.4536088140038, 21.86087709014464),
            #                                   (20.114616870880127, 17.838438272476196, 27.890840008854866),
            #                                   (0.9657187024954271, -0.2595264844008965, 0.005778541776276807)]
            # # tornado
            # pvwidget[0, 0].camera_position = [(-80.1463955070621, -13.21384849813626, 35.76683437307932),
            #                                   (28.435187816619873, 38.87123107910156, 30.663528442382812),
            #                                   (-0.005232506733851609, 0.10831062044128124, 0.9941033298268872)]
            # pvwidget[0, 1].camera_position = [(-80.1463955070621, -13.21384849813626, 35.76683437307932),
            #                                   (28.435187816619873, 38.87123107910156, 30.663528442382812),
            #                                   (-0.005232506733851609, 0.10831062044128124, 0.9941033298268872)]
            # # benard
            # pvwidget[0, 0].camera_position = [(152.5540105346529, -92.68101707874963, 59.95585766901925),
            #                                   (102.41400909423828, 15.50516972830519, 47.394901275634766),
            #                                   (-0.09572746744128804, 0.07091227009602619, 0.9928784930328108)]
            # pvwidget[0, 1].camera_position = [(152.5540105346529, -92.68101707874963, 59.95585766901925),
            #                                   (102.41400909423828, 15.50516972830519, 47.394901275634766),
            #                                   (-0.09572746744128804, 0.07091227009602619, 0.9928784930328108)]
            # # cylinder
            # pvwidget[0, 0].camera_position = [(32.09943574801782, 270.9922387983474, -192.90868512760406),
            #                                   (98.96080662907674, 67.47426327699065, 37.450296361257536),
            #                                   (0.2011556573285153, 0.7623366243331344, 0.6151254121925828)]
            # pvwidget[0, 1].camera_position = [(32.09943574801782, 270.9922387983474, -192.90868512760406),
            #                                   (98.96080662907674, 67.47426327699065, 37.450296361257536),
            #                                   (0.2011556573285153, 0.7623366243331344, 0.6151254121925828)]
            # # 5cp
            # pvwidget[0, 0].camera_position = [(104.29602377612932, 55.81480994809339, -31.593464857709467),
            #                                   (23.915099143981934, 38.04805564880371, 19.034567683935165),
            #                                   (0.10388737447217616, 0.8753831629651112, 0.4721352893212659)]
            # pvwidget[0, 1].camera_position = [(104.29602377612932, 55.81480994809339, -31.593464857709467),
            #                                   (23.915099143981934, 38.04805564880371, 19.034567683935165),
            #                                   (0.10388737447217616, 0.8753831629651112, 0.4721352893212659)]

            pvwidget[0, 0].add_mesh(mesh, style='wireframe', color='white')
            vertexID = np.where(clusterLabel == i)[0]
            pc = mesh.points[vertexID]

            # convert to patch
            faces = self.convertpc2patch(vertexID)
            patch = pv.PolyData(mesh.points, faces)

            pvwidget[0, 0].add_points(pc, color='red', point_size=8)
            # pvwidget[0, 1].add_points(pc, color='red', point_size=8)
            pvwidget[0, 1].add_mesh(patch, color='red')
            pvwidget.show()
            time.sleep(2.0)

    def convertpc2patch(self, vertexID):
        """
        convert the point cloud to patch
        :return: patch's faces, in pyvista format (3, a, b, c)
        """
        t_mesh = trimesh.Trimesh(self.mesh.points, self.mesh.faces.reshape(-1, 4)[:, 1:])
        vertex_faces = t_mesh.vertex_faces

        selected_faceIndex = vertex_faces[vertexID].flatten()
        selected_faceIndex = selected_faceIndex[selected_faceIndex != -1]
        selected_faces = t_mesh.faces[selected_faceIndex]

        faces = np.hstack([np.ones((selected_faces.shape[0], 1), dtype=int) * 3, selected_faces]).flatten()
        return faces

    def get_surfpatch(self):
        """
        get the surface patch of the mesh by the vertex classification result.
        patch ID corresponds to the vertex class number.
        :return: list, each element is a trimesh object.
        """
        t_mesh = trimesh.Trimesh(self.mesh.points, self.mesh.faces.reshape(-1, 4)[:, 1:])
        vertex_faces = t_mesh.vertex_faces

        patch_cls = np.unique(self.clusterLabel)
        from collections import OrderedDict
        self.patchDict = OrderedDict()
        self.patch_hks = OrderedDict()
        for pcls in patch_cls:
            vertexID = np.where(self.clusterLabel == pcls)[0]
            selected_faceID = vertex_faces[vertexID].flatten()
            selected_faceID = selected_faceID[selected_faceID != -1]
            patch = t_mesh.submesh([selected_faceID])[0]
            self.patchDict[pcls] = patch
            self.patch_hks[pcls] = self.hks[vertexID].mean(axis=0)

    def get_patchInfo(self):
        """
        get the patch information
        :return: dict, key is the patch ID, value is the patch object.
        """
        return self.patchDict, self.patch_hks, self.clusterLabel


def generate_random_hex_color():
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color

