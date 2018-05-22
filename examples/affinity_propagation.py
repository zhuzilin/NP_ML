from np_ml import AffinityPropagation
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

if __name__ == '__main__':
    centers = [[1, 1], [-1, -1], [1, -1]]
    x, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                                random_state=0)

    ap = AffinityPropagation()
    label = ap.fit(x, preference=-50)
    print("cluster centers: ")
    print(ap.centers)
    plt.scatter(x[:, 0], x[:, 1], c=label)
    plt.show()