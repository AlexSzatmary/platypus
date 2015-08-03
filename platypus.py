import brewer2mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import string


def make_markers():
    markers = '.,ov^<>1234sp*hH+xDd|_'
    good_markers = [2, 4, 11, 12, 13, 15, 16, 17, 18, 19, 20]
    liney_markers = [7, 8, 9, 10, 16, 17, 20, 21]
    L_marker = []
    L_edgewidth = []
    for i in good_markers:
        if i in liney_markers:
            L_edgewidth.append(3.)
        else:
            L_edgewidth.append(1.)
        L_marker.append(markers[i])
    return (L_marker, L_edgewidth)


FORMAT = '.pdf'
set1 = brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors
set1.append((0., 0., 0.1))
set2 = set1[:5] + set1[6:]  # set2 is like set1 but without yellow
set3 = set2[:-1]  # set3 is like set2 but without black
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
            return set3[(j - 1) % len(set3)]
    return f
set0_color_f = setn_color_f(0)
#COLORS = 'bgrcmykw'.replace('w', '')


class Figure(object):
    def __init__(self, axes=None, figsize=None, panesize=None,
                 subplot=None, legend_bbox=None,
                 legend_outside=False, xlabelpad=None, ylabelpad=None):
        self.axes = axes

        if subplot is None:
            subplot = (1, 1, 1)
            wspace = 0.
            hspace = 0.
        else:
            if subplot[1] > 1:
                wspace = (1. - self.axes[2]) / self.axes[2]
            else:
                wspace = 0.
            if subplot[0] > 1:
                hspace = (1. - self.axes[3]) / self.axes[3]
            else:
                hspace = 0.

        if figsize is None:
            figsize = (panesize[0] * subplot[1], panesize[1] * subplot[0])

        self.fig = plt.figure(figsize=figsize)

        if legend_bbox is None:
            self.legend_bbox = (1.05, 0.5)
        else:
            self.legend_bbox = legend_bbox

        self.legend_outside = legend_outside
        if self.legend_outside:
            self.figlegend = plt.figure(figsize=panesize)

        if subplot:
            self.fig.subplots_adjust(
                left=(self.axes[0] / subplot[1]),
                bottom=(self.axes[1] / subplot[0]),
                right=(1. - (1. - self.axes[0] - self.axes[2]) / subplot[1]),
                top=(1. - (1. - self.axes[1] - self.axes[3]) / subplot[0]),
                wspace=wspace, hspace=hspace)
            self.fig.add_subplot(*subplot)
        else:
            self.fig.add_axes(self.axes)
        self.xlabelpad = xlabelpad
        self.ylabelpad = ylabelpad
        self.set_tick_font()
        self.clean_axes()

    def clean_axes(self):
        ax = self.fig.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    def set_tick_font(self):
        ax = self.fig.gca()        
        for label in ax.get_xticklabels():
            label.set_font_properties(self.tick_font_properties)
        for label in ax.get_yticklabels():
            label.set_font_properties(self.tick_font_properties)

    def set_xint(self):
        ax = self.fig.gca()
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10],
                                          integer=True))

    def set_yint(self):
        ax = self.fig.gca()
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10],
                                          integer=True))


    def plot(self, *args, **kwargs):
        ax = self.fig.gca()
        lines = ax.plot(*args, **kwargs)

    def set_xlabel(self, *args, labelpad=None, **kwargs):
        if 'fontproperties' not in kwargs:
            kwargs['fontproperties'] = self.font_properties
        ax = self.fig.gca()
        if labelpad is None:
            labelpad = self.xlabelpad
        self.fig.gca().set_xlabel(*args, labelpad=labelpad, **kwargs)

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

    def add_subplot(self, *args):
        self.fig.add_subplot(*args)
        self.clean_axes()
        self.set_tick_font()

    def set_AB_labels(self):
        for (j, ax) in enumerate(self.fig.get_axes()):
            ax.text(-0.25, 0.975, string.ascii_uppercase[j],
                     font_properties=self.AB_font_properties,
                     transform=ax.transAxes)

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

    def savefig(self, file_name, my_format=FORMAT, path=None, **kwargs):
        ax = self.fig.gca()
        if path is not None:
            out_file_path = os.path.join(path, file_name + my_format)
        else:
            out_file_path = os.path.join(file_name + my_format)
        self.fig.savefig(out_file_path, **kwargs)
        if self.legend_outside:
            if path is not None:
                out_file_name_legend = os.path.join(
                    path, file_name + '-legend' + my_format)
            else:
                out_file_name_legend = path, file_name + '-legend' + my_format
            self.figlegend.savefig(out_file_name_legend + my_format, **kwargs)


class Print(Figure):
    style = 'print'
    font_properties = matplotlib.font_manager.FontProperties(
        family='Times', size=10)
    tick_font_properties = font_properties.copy()
    tick_font_properties.set_size(8)
    AB_font_properties = matplotlib.font_manager.FontProperties(
        family='Helvetica', size=14)

    def __init__(self, axes=[0.25,  0.25, 0.65,  0.65],
                 panesize=(3., 3.),
                 xlabelpad=None, ylabelpad=None,
                 **kwargs):
        super().__init__(
            axes=axes, panesize=panesize,
            xlabelpad=xlabelpad, ylabelpad=ylabelpad,
            **kwargs)

    def clean_axes(self):
        super().clean_axes()
        ax = self.fig.gca()
        ax.spines['left']._linewidth = 0.5
        ax.spines['bottom']._linewidth = 0.5


class Poster(Figure):
    style = 'poster'
    font_properties = matplotlib.font_manager.FontProperties(
        family='Palatino', size=20)
    tick_font_properties = font_properties.copy()
    AB_font_properties = matplotlib.font_manager.FontProperties(
        family='Helvetica', size=14)

    def __init__(self, axes=[0.2, 0.2, 0.75, 0.75],
                 panesize=(7., 7.), 
                 xlabelpad=None, ylabelpad=None,
                 **kwargs):
        super().__init__(
            axes=axes, panesize=panesize,
            xlabelpad=xlabelpad, ylabelpad=ylabelpad,
            **kwargs)


class Projector(Figure):
    style = 'projector'
    font_properties = matplotlib.font_manager.FontProperties(
        family='Helvetica', size='x-large')
    tick_font_properties = font_properties.copy()
    AB_font_properties = matplotlib.font_manager.FontProperties(
        family='Helvetica', size=14)

    def __init__(self, axes=[0.14, 0.14, 0.8, 0.8],
                 panesize=(8., 6.), 
                 xlabelpad=None, ylabelpad=None,
                 **kwargs):
        super().__init__(
            axes=axes, panesize=panesize,
            xlabelpad=xlabelpad, ylabelpad=ylabelpad,
            **kwargs)


class RSC(Print):
    def __init__(self, axes=[0.25,  0.25, 0.65,  0.65],
                 panesize=(3.25, 3.25),
                 **kwargs):
        super().__init__(axes=axes, panesize=panesize,
                         **kwargs)


d_fig_cls = {'print': Print, 'RSC': RSC, 'poster': Poster,
             'projector': Projector}

def figure(style='print', **kwargs):
    return d_fig_cls[style](**kwargs)


def _plot(
    plot_callback,
    fig=None,
    # Figure object parameters
    axes=None, figsize=None, style='print', subplot=None, legend_bbox=None,
    legend_outside=False,
    # End figure object parameters
    title='', xlabel='', ylabel='', L_legend=None,
    xlog=None, xlim=None, xint=None, ylog=None, ylim=None, yint=None,
    path=None, file_name='', my_format=FORMAT, tight=False,
    **kwargs):
    '''
    Helper for easy plotting functions
    '''

    if fig is None:
        if (L_legend is not None) and not legend_outside and subplot is None:
            subplot = (1, 2, 1)
        fig = figure(axes=axes, figsize=figsize, style=style, subplot=subplot,
                     legend_bbox=legend_bbox, legend_outside=legend_outside)

    ax = fig.fig.gca()

    plot_callback(fig, **kwargs)

    if xlog:
        ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylog:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if xlabel:
        fig.set_xlabel(xlabel)
    if ylabel:
        fig.set_ylabel(ylabel)

    if L_legend:
        fig.legend(L_legend)

    if title:
        fig.title(title)

    fig.fig.canvas.draw()    

    if tight:
        bbox_inches = 'tight'
    else:
        bbox_inches = None

    if file_name:
        fig.savefig(file_name, my_format, path=path,
                    bbox_inches=bbox_inches)
    return fig



def _multi_plot_helper(fig, L_x, L_y,
    L_marker=None, L_markeredgewidth=None, L_linestyle=None, color_f=set3_color_f,
    **kwargs):
    for (i, (x, y)) in enumerate(zip(L_x, L_y)):
        if L_marker:
            kwargs['marker'] = L_marker[i]
        if L_markeredgewidth:
            kwargs['markeredgewidth'] = L_markeredgewidth[i]
        if L_linestyle:
            kwargs['linestyle'] = L_linestyle[i]
        line = fig.plot(x, y, color=color_f(i), **kwargs)



def multi_plot(L_x, L_y, **kwargs):
    '''
    Easy default one-line function interface for making plots
    '''
    def my_callback(fig, **kwargs):
        return _multi_plot_helper(fig, L_x, L_y, **kwargs)
    fig = _plot(my_callback, **kwargs)
    return fig


def _boxplot_helper(
    fig, x,
    boxprops={'color': 'black'}, capprops={'color': 'black'},
    flierprops={'color': 'black'}, meanprops={'color': 'black'},
    medianprops={'color': 'black'}, whiskerprops={'color': 'black'},
    **kwargs):
    ax = fig.fig.gca()
    ax.boxplot(
        x, boxprops=boxprops, capprops=capprops,
        flierprops=flierprops, meanprops=meanprops,
        medianprops=medianprops, whiskerprops=whiskerprops,
        **kwargs)


def boxplot(x, notch=False, sym='k.', vert=False, axes=None, **kwargs):
    '''
    Easy default one-line function interface for making boxplots
    '''
    if axes is None:
        axes = [0.4,  0.25, 0.55,  0.65]
    def my_callback(fig, **kwargs):
        return _boxplot_helper(fig, x, **kwargs)
    fig = _plot(my_callback, axes=axes, notch=notch, sym=sym, vert=vert,
                **kwargs)
    return fig
