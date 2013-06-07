import brewer2mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def make_markers():
    markers = '.,ov^<>1234sp*hH+xDd|_'
    good_markers = [2, 4, 11, 12, 13, 15, 16, 17, 18, 19, 20]
    liney_markers = [7, 8, 9, 10, 16, 17, 20, 21]
    l_markers = []
    for i in good_markers:
        marker = {'marker': markers[i]}
        if i in liney_markers:
            marker['edgewidth'] = 3.
        else:
            marker['edgewidth'] = 1.
        l_markers.append(marker)
    return l_markers


FORMAT = '.pdf'
set1 = brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors
set1.append((0., 0., 0.1))
COLORS = 'bgrcmykw'.replace('w', '')


class figure(object):
    def __init__(self, axes=None, figsize=(7.5, 6.),
                 style='print', subplot=None):
        self.fig = plt.figure(figsize=(7.5, 6.))
        if subplot:
            self.legend_bbox = (1.2, 0.5)
        else:
            self.legend_bbox = None
        if axes:
            self.axes = axes
        else:
            self.axes = [0.105, 0.12, 0.6, 0.85]
#        self.fig.add_axes(self.axes)
        if subplot:
            self.fig.subplots_adjust(left=self.axes[0], bottom=self.axes[1],
                                     right=(self.axes[0] + self.axes[2]),
                                     top=(self.axes[1] + self.axes[3]))
            self.fig.add_subplot(*subplot)
        if style == 'projector':
            self.font_properties = matplotlib.font_manager.FontProperties(
                family='Helvetica', size='x-large')
        else:
            self.font_properties = matplotlib.font_manager.FontProperties(
                family='Times')
        self.set_ticks()
        self.clean_axes()

    def clean_axes(self):
        ax = self.fig.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left']._linewidth = 0.5
        ax.spines['bottom']._linewidth = 0.5
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    def set_ticks(self, xlocator=matplotlib.ticker.LinearLocator()):
        ax = self.fig.gca()
        ax.xaxis.set_major_locator(xlocator)
        ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator())
        ax.set_xticklabels(ax.get_xticks(),
                           fontproperties=self.font_properties)
        ax.set_yticklabels(ax.get_yticks(),
                           fontproperties=self.font_properties)
        self.fig.canvas.draw()

    def plot(self, *args, **kwargs):
        ax = self.fig.gca()
        ax.plot(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        if 'fontproperties' not in kwargs:
            kwargs['fontproperties'] = self.font_properties
        self.fig.gca().set_xlabel(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        if 'fontproperties' not in kwargs:
            kwargs['fontproperties'] = self.font_properties
        self.fig.gca().set_ylabel(*args, **kwargs)

    def legend(self, *args, **kwargs):
        if 'fontproperties' not in kwargs:
            kwargs['prop'] = self.font_properties
        if 'bbox_to_anchor' not in kwargs and self.legend_bbox is not None:
            kwargs['bbox_to_anchor'] = self.legend_bbox
            if 'loc' not in kwargs:
                kwargs['loc'] = 10
        self.fig.gca().legend(*args, **kwargs)

    def quiver(self, arr, color=None, scaled=True, ax=None):
        """
        Quiver plot of array data, where column 0 is x1, 1 is y1, 2 is x2, and
        3 is y2. x and y are scaled spatially.
        """
        self.fig.gca().axis
        plt.quiver(arr[:, 0], arr[:, 1], arr[:, 2] - arr[:, 0],
                   arr[:, 3] - arr[:, 1],
                   figure=self.fig,
                   width=0.005, angles='xy',
                   scale_units='xy', scale=1., headaxislength=5,
                   color=color)
        if scaled:
            self.fig.gca().axis('scaled')
        if not ax:
            ax = self.fit_quiver_axis(arr)
            self.fig.gca().set_xlim(ax[0], ax[1])
            self.fig.gca().set_ylim(ax[2], ax[3])

    def fit_quiver_axis(self, arr):
        xmin = min(np.min(arr[:, 0]), np.min(arr[:, 2]))
        xmax = max(np.max(arr[:, 0]), np.max(arr[:, 2]))
        ymin = min(np.min(arr[:, 1]), np.min(arr[:, 3]))
        ymax = max(np.max(arr[:, 1]), np.max(arr[:, 3]))
        return [xmin, xmax, ymin, ymax]
