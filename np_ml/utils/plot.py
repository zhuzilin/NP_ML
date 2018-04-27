import matplotlib.pyplot as plt

def plot(x, color, **kwargs):
    # can only do 2D plot right now
    assert(x.shape[-1] == 2)
    if 'accuracy' in kwargs:
        accuracy = kwargs['accuracy']
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=color)
    if 'title' in kwargs:
        plt.suptitle(kwargs['title'])
    if 'accuracy' in kwargs:
        plt.title("Accuracy: %.1f%%" % (kwargs['accuracy']*100), fontsize=10)
    plt.show()