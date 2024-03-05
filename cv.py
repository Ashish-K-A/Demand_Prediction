# cv.py

from types import GeneratorType
import pandas as pd

from sklearn.model_selection._split import TimeSeriesSplit
from sklearn.utils.validation import _deprecate_positional_args

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

import numpy as np

class NestedCV(TimeSeriesSplit):
    """
    parameters
    ----------
    n_test_folds: int
        number of folds to be used as testing at each iteration.
        by default, 1.
    """
    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, max_train_size=None, n_test_folds=1):
        super().__init__(n_splits, 
                         max_train_size=max_train_size)
        self.n_test_folds=n_test_folds

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + self.n_test_folds
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater than the number of samples: {1}.").format(n_folds,n_samples))
        indices = np.arange(n_samples)
        fold_size = (n_samples // n_folds)
        test_size = fold_size * self.n_test_folds # test window
        test_starts = range(fold_size + n_samples % n_folds,
                            n_samples-test_size+1, fold_size) # splits based on fold_size instead of test_size
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (indices[test_start - self.max_train_size:test_start],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])

if __name__ == "__main__":
    # load dataset
    data = pd.read_csv("Data/X_train.csv")
    data["date"] = pd.to_datetime(data["date"])

    # nested cv
    k = 3
    cv = NestedCV(k)
    splits = cv.split(data, "date")

    # check return type
    assert isinstance(splits, GeneratorType)

    # check return types, shapes, and data leaks
    count = 0
    for train, validate in splits:
        # types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(validate, pd.DataFrame)

        # shape
        assert train.shape[1] == validate.shape[1]

        # data leak
        assert train["date"].max() <= validate["date"].min()

        count += 1

    # check number of splits returned
    assert count == k