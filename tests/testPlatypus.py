import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import platypus


@image_comparison(baseline_images=['plot0'], extensions=['pdf', 'png'])
def test_0():
    fig = platypus.Figure()
    x = np.linspace(0, 2*np.pi, 100)
    y = 2*np.sin(x)
    fig.plot(x, y)
    fig.set_xlabel('foo')
    fig.set_ylabel('bar')


@image_comparison(baseline_images=['test_AB_labels'],
                  extensions=['pdf', 'png'])
def test_AB_labels():
    fig = platypus.Figure(subplot=(2, 2, 1))
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
