import numpy as np
import matplotlib
from .image_comparison import image_comparison
import matplotlib.pyplot as plt
from .. import platypus

def test_nose():
    pass

@image_comparison(baseline_images=['plot0'], extensions=['pdf', 'png'])
def test_0():
    fig = platypus.figure()
    x = np.linspace(0, 2*np.pi, 100)
    y = 2*np.sin(x)
    fig.plot(x, y)
    fig.set_xlabel('foo')
    fig.set_ylabel('bar')


@image_comparison(baseline_images=['test_AB_labels'],
                  extensions=['pdf', 'png'])
def test_AB_labels():
    fig = platypus.figure(subplot=(2, 2, 1))
    x = np.linspace(0, 2*np.pi, 100)
    y = 2*np.sin(x)
    fig.plot(x, y)
    fig.set_xlabel('foo')
    fig.set_ylabel('bar')
    fig.add_subplot(*(2, 2, 2))
    fig.plot(x, y + 1.)
    fig.set_xlabel('foo')
    fig.set_ylabel('bar')
    fig.add_subplot(*(2, 2, 3))
    fig.plot(x, y - 1.)
    fig.set_xlabel('foo')
    fig.set_ylabel('bar')
    fig.set_AB_labels()

@image_comparison(baseline_images=['test_multi_plot'],
                  extensions=['pdf', 'png'])
def test_multi_plot():
    x = np.linspace(0, 2*np.pi, 100)
    y = 2*np.sin(x)
    fig = platypus.multi_plot([x], [y], xlabel='foo', ylabel='bar')
