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
set2 = set1[:5] + set1[6:] # set2 is like set1 but without yellow
set3 = set2[:-1] # set3 is like set2 but without black
BLACK = set2[-1]

def loop_list(L):
    return lambda i: L[i % len(L)]

set1_color_f = loop_list(set1)
set2_color_f = loop_list(set2)
set3_color_f = loop_list(set3)
def setn_color_f(k):
    def f(j):
        if j == k:
            return BLACK
        elif j < k:
            return set3[j % len(set3)]
        elif j > k:
            return set3[(j + 1) % len(set3)]
    return f
#COLORS = 'bgrcmykw'.replace('w', '')


class Figure(object):
    def __init__(self, axes=None, figsize=None,
                 style='print', subplot=None, legend_bbox=None,
                 legend_outside=False):
        self.style = style

        if subplot is None:
            subplot = (1, 1, 1)

        if figsize is None:
            if self.style == 'print':
                panesize = (3., 3.)
            elif style == 'projector':
                panesize = (8., 6.)
            figsize = (panesize[0] * subplot[1],
                       panesize[1] * subplot[0])
        self.fig = plt.figure(figsize=figsize)

        if legend_bbox is None:
            self.legend_bbox = (1.05, 0.5)
        else:
            self.legend_bbox = legend_bbox
        if axes:
            self.axes = axes
        else:
#            self.axes = None
            if self.style == 'print':
                self.axes = [0.25,  0.25, 0.6,  0.6]
            elif self.style == 'projector':
                self.axes = [0.14, 0.1, 0.8, 0.8]

        self.legend_outside = legend_outside
        if self.legend_outside:
            self.figlegend = plt.figure(figsize=panesize)

        if subplot:
            self.fig.subplots_adjust(
                left=(self.axes[0] / subplot[1]),
                bottom=(self.axes[1] / subplot[0]),
                right=(1. - (1. - self.axes[0] - self.axes[2]) / subplot[1]),
                top=(1. - (1. - self.axes[1] - self.axes[3]) / subplot[0]))
            self.fig.add_subplot(*subplot)
        else:
            self.fig.add_axes(self.axes)
        if self.style == 'projector':
            self.font_properties = matplotlib.font_manager.FontProperties(
                family='Helvetica', size='x-large')
        else:
            self.font_properties = matplotlib.font_manager.FontProperties(
                family='Times', size=10)
        self.set_ticks()
        self.clean_axes()

    def clean_axes(self):
        ax = self.fig.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if self.style == 'print':
            ax.spines['left']._linewidth = 0.5
            ax.spines['bottom']._linewidth = 0.5
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    def set_ticks(self, xlocator=None, ylocator=None, xint=True, yint=True):
        ax = self.fig.gca()
        xlog = ax.get_xscale() == 'log'
        ylog = ax.get_yscale() == 'log'
        # if xlocator is None:
        #     xlocator = matplotlib.ticker.AutoLocator()
        # if ylocator is None:
        #     ylocator = matplotlib.ticker.AutoLocator()
        if xlocator is not None:
            ax.xaxis.set_major_locator(xlocator)
        if ylocator is not None:
            ax.yaxis.set_major_locator(ylocator)

        if xlog:
            pass
        elif xint:
            ax.set_xticklabels(
                [int(xx) if xx.is_integer() else xx for xx in ax.get_xticks()],
                fontproperties=self.font_properties)
        else:
            ax.set_xticklabels(ax.get_xticks(),
                               fontproperties=self.font_properties)

        if ylog:
            pass
        elif yint:
            ax.set_yticklabels(
                [int(xx) if xx.is_integer() else xx for xx in ax.get_yticks()],
                fontproperties=self.font_properties)
        else:
            ax.set_yticklabels(ax.get_yticks(),
                fontproperties=self.font_properties)
        self.fig.canvas.draw()

    def plot(self, *args, **kwargs):
        ax = self.fig.gca()
        lines = ax.plot(*args, **kwargs)

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
        if not self.legend_outside:
            if 'bbox_to_anchor' not in kwargs and self.legend_bbox is not None:
                kwargs['bbox_to_anchor'] = self.legend_bbox
                if 'loc' not in kwargs:
                    kwargs['loc'] = 'center left'
            self.my_legend = self.fig.gca().legend(*args, **kwargs)
        else:
            kwargs['loc'] = 'center'
            self.my_legend = self.figlegend.gca().legend(
                *([self.fig.gca().lines] + list(args)), **kwargs)
            for q in self.figlegend.gca().get_children():
                if not issubclass(q.__class__, matplotlib.legend.Legend):
                    q.set_visible(False)

    def title(self, *args, **kwargs):
        if 'fontproperties' not in kwargs:
            kwargs['fontproperties'] = self.font_properties
        self.fig.gca().set_title(*args, **kwargs)

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

    def savefig(self, file_name, format, **kwargs):
        self.set_ticks()
        ax = self.fig.gca()
        self.fig.savefig(file_name + format, **kwargs)
        if self.legend_outside:
            self.figlegend.savefig(file_name + '-legend' + format, **kwargs)


def multi_plot(
    L_x, L_y,
    fig=None, file_name='', my_format=FORMAT,
    color_f=set3_color_f,
    L_legend=None, legend_outside=False,
    title='',
    xlog=False, xlim=None,
    ylog=False, ylim=None,
    xlabel='', ylabel='',
    xint=False, yint=False, style=None):
    '''
    Easy default one-line function interface for making plots
    '''

    if fig == None:
        if (L_legend is not None) and not legend_outside:
            subplot = (1, 2, 1)
        else:
            subplot = (1, 1, 1)
        if style is None:
            style = 'print'
        fig = Figure(subplot=subplot, style=style,
                     legend_outside=legend_outside)

    ax = fig.fig.gca()

    for (i, (x, y)) in enumerate(zip(L_x, L_y)):
        fig.plot(x, y, color=color_f(i))            
    fig.set_ticks(xint=xint, yint=yint)
    fig.fig.canvas.draw()    
    if xlabel:
        fig.set_xlabel(xlabel)
    if ylabel:
        fig.set_ylabel(ylabel)

    if xlog:
        ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylog:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if L_legend:
        fig.legend(L_legend)

    if title:
        fig.title(title)

    if file_name:
        fig.savefig(file_name, my_format)
    return fig
