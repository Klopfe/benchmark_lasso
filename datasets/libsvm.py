from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from sklearn.preprocessing import StandardScaler
    import numpy as np


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': ["bodyfat", "leukemia", "mnist", "rcv1.binary"],
    }

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset

    def get_data(self):
        X, y = fetch_libsvm(self.dataset)

        if self.dataset == "mnist":
            X = X.toarray()
            X = StandardScaler().fit_transform(X)
            y -= np.mean(y)
        data = dict(X=X, y=y)

        return X.shape[1], data
