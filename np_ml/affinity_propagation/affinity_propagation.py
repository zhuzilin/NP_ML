"""
    The implementation is a simpler version of sklearn implementation
    which is different from the one in wikipedia
"""
import numpy as np
import scipy


class AffinityPropagation:
    def __init__(self):
        self.centers = None

    def fit(self, x, damp=0.5, max_epoch=200, convergence_iter=15, preference=None):
        n_sample = x.shape[0]
        assert n_sample > 1, "too little data"
        centers = np.zeros(x.shape[0])
        s = -scipy.spatial.distance.cdist(x, x)**2  # caution! the cdist is ||x1 - x_2||
        if preference is None:
            preference = np.median(s)
        s.flat[::(x.shape[0] + 1)] = preference
        r = np.zeros(s.shape)
        a = np.zeros(s.shape)
        epoch = 0
        same_iter = 0
        ind = np.arange(n_sample)
        while epoch < max_epoch and same_iter < convergence_iter:
            tmp = s + a
            maxa_ind = np.argmax(tmp, axis=1)
            maxa = tmp[ind, maxa_ind]
            tmp[ind, maxa_ind] = -np.inf
            maxa_2 = np.max(tmp, axis=1)

            r_new = s - maxa[:, None]
            r_new[ind, maxa_ind] = s[ind, maxa_ind] - maxa_2
            r = damp * r + (1. - damp) * r_new

            a_new = np.maximum(r, 0)
            a_new.flat[::n_sample+1] = r.flat[::n_sample+1]

            a_new -= np.sum(a_new, axis=0)
            da = np.diag(a_new).copy()
            a_new = a_new.clip(0, np.inf)
            a_new.flat[::n_sample+1] = da
            a = damp*a - (1.-damp)*a_new

            centers_new = (np.diag(r)+np.diag(a)) > 0
            if 0 < np.sum(centers_new) < n_sample and (centers == centers_new).all():
                same_iter += 1
            else:
                centers = centers_new
                same_iter = 0
            epoch += 1
        self.centers = x[centers, :]
        # after find centers, use knn
        s.flat[::n_sample+1] = 0.  # change the s from
        label = s[:, centers].argmax(axis=1)
        return label

    def predict(self):
        dist_matrix = scipy.spatial.distance.cdist(x, self.centers)
        label = np.argmin(dist_matrix, axis=1)
        return label
