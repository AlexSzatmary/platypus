import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def label_curves(axis=None):
    """
    Automatically put legend-like labels right on curves

    This function was originally written by Jan Kuiken as an
    [answer](http://stackoverflow.com/a/17000662/400793) to a question I had
    asked on Stack Overflow
    """

    if axis == None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    print Nlines

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((Nlines, N, N), dtype=np.float)    

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin,ymin]) / ([xmax-xmin, ymax-ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot        
        mask = (xy[:,0] >= 0) & (xy[:,0] < N) & (xy[:,1] >= 0) & (xy[:,1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0 
    # don't use the borders
    ws[:,0]   = 0
    ws[:,N-1] = 0
    ws[0,:]   = 0  
    ws[N-1,:] = 0  

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N/5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(Nlines, dtype=np.float)
        w[l] = 0.5

        # calculate a field         
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
        # plt.figure()
        # plt.imshow(p, interpolation='nearest')
        # plt.title(axis.lines[l].get_label())

        pos = np.argmax(p)  # note, argmax flattens the array first 
        best_x, best_y =  (pos / N, pos % N) 
        x = xmin + (xmax-xmin) * best_x / N       
        y = ymin + (ymax-ymin) * best_y / N       

        print axis.lines[l].get_label(), x, y
        axis.text(x, y, axis.lines[l].get_label(), 
                  horizontalalignment='center',
                  verticalalignment='center')

def example_legend():
    plt.clf()
    x = np.linspace(0, 1, 101)
    y1 = np.sin(x * np.pi / 2)
    y2 = np.cos(x * np.pi / 2)
    plt.plot(x, y1, label='sin')
    plt.plot(x, y2, label='cos')
    plt.title('Legend')
    plt.legend()

def example_inline():
    plt.clf()
    x = np.linspace(0, 1, 101)
    y1 = np.sin(x * np.pi / 2)
    y2 = np.cos(x * np.pi / 2)
    plt.plot(x, y1, label='sin')
    plt.plot(x, y2, label='cos')
    plt.text(0.08, 0.2, 'sin')
    plt.text(0.9, 0.2, 'cos')
    plt.title('Inline')

def example_unlabeled():
#    plt.clf()
    x = np.linspace(0, 1, 101)
    y1 = np.sin(x * np.pi / 2)
    y2 = np.cos(x * np.pi / 2)
    plt.plot(x, y1, 'b', label='sin')
    plt.plot(x, y2, 'r', label='cos')
    # plt.text(0.08, 0.2, 'sin')
    # plt.text(0.9, 0.2, 'cos')
#    plt.title('Unlabeled')
#    label_curves()
#    plt.show()

def example_from_stack():
    plt.close('all')

    x = np.linspace(0, 1, 101)
    y1 = np.sin(x * np.pi / 2)
    y2 = np.cos(x * np.pi / 2)
    y3 = x * x
    plt.plot(x, y1, 'b', label='blue')
    plt.plot(x, y2, 'r', label='red')
    plt.plot(x, y3, 'g', label='green')
    label_curves()
#    plt.show()
