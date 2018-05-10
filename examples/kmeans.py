from sklearn.datasets import make_blobs
from np_ml import KMeans
from np_ml.utils import plot

if __name__ == '__main__':
    n_samples = 1500
    random_state = 170
    x, y = make_blobs(n_samples=n_samples, random_state=random_state)
    print(type(x))
    print(x.shape)
    # Incorrect number of clusters
    y_pred = KMeans(k=3).fit(x, detailed=True)
    plot(x, y_pred, title="KMeans")