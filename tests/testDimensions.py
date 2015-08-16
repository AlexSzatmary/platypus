import numpy as np
import matplotlib
from .image_comparison import image_comparison
import matplotlib.pyplot as plt
from .. import platypus


def helper_dimensions(style):
    x = 90000. / 7. * np.linspace(0, 2*np.pi, 100)
    y = 9000. / 3. * 2*np.sin(x)
    fig = platypus.multi_plot(
        [x], [y], xlabel=r'$\frac{dx}{dt}$', ylabel=r'$\frac{dy}{dt}$',
        xlim=(0., 9e4), xint=False,
        style=style)

@image_comparison(
    baseline_images=['test_dimensions_' + style
                     for style in platypus.d_fig_cls.keys()],
    extensions=['pdf', 'png'])
def test_dimensions():
    for style in platypus.d_fig_cls.keys():
        helper_dimensions(style)
        
