import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot(x, y, **kwargs):
    # can only do 2D plot right now
    assert(x.shape[-1] == 2)
    color = (y+2) / 5  # TODO: not hard code color
    if 'accuracy' in kwargs:
        accuracy = kwargs['accuracy']
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=color)
    if 'title' in kwargs:
        plt.suptitle(kwargs['title'])
    if 'accuracy' in kwargs:
        plt.title("Accuracy: %.1f%%" % (kwargs['accuracy']*100), fontsize=10)
    plt.show()
    
def plot_boundary(model, x, y, **kwargs):
    assert(x.shape[-1] == 2)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    if 'h' in kwargs:
        h = kwargs['h']
    else:
        h = 0.1
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(x_grid.shape)
    plt.figure()
    plt.pcolormesh(x_grid, y_grid, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(x_grid.min(), x_grid.max())
    plt.ylim(y_grid.min(), y_grid.max())
    
    if 'title' in kwargs:
        plt.suptitle(kwargs['title'])
    if 'accuracy' in kwargs:
        plt.title("Accuracy: %.1f%%" % (kwargs['accuracy']*100), fontsize=10)
    plt.show()