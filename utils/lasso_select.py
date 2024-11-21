import random

import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, canvas, alpha_other=0.9, lasso_active=True, signal=None):
        # self.canvas = ax.figure.canvas
        self.click_ind = None
        self.canvas = canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.lasso_active = lasso_active
        self.signal = signal

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

        ax.figure.canvas.mpl_connect('button_press_event', self.onpick)

    def onselect(self, verts):
        if self.lasso_active == True:
            path = Path(verts)
            self.ind = np.nonzero(path.contains_points(self.xys))[0]
            self.fc[:, -1] = self.alpha_other
            self.fc[self.ind, -1] = 1
            self.collection.set_facecolors(self.fc)
            self.collection.set_edgecolors(["black" if i in self.ind else "white" for i in range(self.Npts)])
            self.canvas.draw_idle()
            # ============render version================
            if self.signal is not None:
                self.signal.emit(True)
                self.signal.emit(False)

    def onpick(self, event):
        if not self.lasso_active:
            self.click_ind = find_nearest_index(self.xys, (event.xdata, event.ydata))
            self.fc[:, -1] = self.alpha_other
            self.fc[self.click_ind, -1] = 1
            self.collection.set_facecolors(self.fc)
            self.collection.set_edgecolors(["red" if i == self.click_ind else "white" for i in range(self.Npts)])
            self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def find_nearest_index(xys, target_coord):
    distances = np.linalg.norm(xys - target_coord, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index


def lassoSelector(fig, ax, canvas, x_embedding, y_pred, colormap=None, lasso_active=True,signal=None) -> plt.axes:
    """
    Select points in a matplotlib figure.
    Selected points are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.
    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).
    Returns
    -------
    ind : list
        List of selected indices.
    """
    if colormap is None:
        cls_num = np.unique(y_pred).shape[0]
        colormap = [generate_random_hex_color() for _ in range(cls_num)]
        colors = [colormap[i] for i in y_pred]

    # ====================== only for experiment ======================
    # if colormap is None:
    #     cls_num = np.unique(y_pred).shape[0]
    #     colormap = [generate_random_hex_color() for _ in range(cls_num)]
    #     colors = ["#ACACAC" for _ in y_pred]
    # =================================================================

    else:
        colors = [colormap[i % len(colormap)] for i in y_pred]

    cls_num = np.unique(y_pred).shape[0]
    pts = ax.scatter(x_embedding[:, 0], x_embedding[:, 1], s=140, c=colors, edgecolors='white')
    selector = SelectFromCollection(ax, pts, canvas, lasso_active=lasso_active, signal=signal)
    canvas.draw()

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            # print(selector.xys[selector.ind])
            print(selector.ind)
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)

    return selector, colors


def generate_random_hex_color():
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color


if __name__ == '__main__':

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    data = np.random.rand(100, 2)

    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    colors = ['red', 'green', 'blue', 'orange', 'purple'] * 20
    pts = ax.scatter(data[:, 0], data[:, 1], s=80, c=colors)
    selector = SelectFromCollection(ax, pts, None)


    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            # print(selector.xys[selector.ind])
            print(selector.ind)
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()


    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    plt.show()
