import numpy as np
import matplotlib.pyplot as plt


class SVM(object):
    """ An easy implementation of SVM algorithm in machine learning.

        A standard SVM algorithm could be represented as:
        argmin 0.5 * (w ^ T * w)
        s.t.   train_yi * (w ^ T * train_xi) >= 1

        If support kernel function, then the the programming problem become:
        argmin 0.5 * (w ^ T * w)
        s.t.   train_yi * (w ^ T * kernel(train_xi)) >= 1

        If soft margin is supported,
        Consider a slack variable slack_i >= 0, then the programming problem become:
        argmin 0.5 * (w ^ T * w) + C * sum( error_function(slack_i) )
        s.t.   train_yi * (w ^ T * train_xi) >= 1 - slack_i
               slack_i >= 0
        C is a constant here (hyper parameter) to measure the softness of margin.
        error_function(x) = 0 if x >= 1; and >= 0 otherwise
            (sometime we set error_function(x) = 0 when x >= 0 and > 0 otherwise).
    """
    def __init__(self, train_x: np.matrix, train_y, kernel=None, is_hard_margin=True, error_function=None):
        """
        Args:
            train_x: a matrix of train features.
            train_y: a vertical vector
            kernel: kernel function
            is_hard_margin: if it is false, support soft margin
            error_function: penalty of points within margin or miss classified.
        """
        if (kernel is not None) or (not is_hard_margin) or (error_function is not None):
            raise NotImplementedError("Kernel and soft margin are not supported now.")
        self._train_x = train_x.shape
        self._train_y = train_y
        self._feature_cnt = train_x.shape
        self._parameter = np.zeros(1, 10)

    def train(self, train_x, train_y):
        pass

    def test(self, test_x, test_y):
        pass

    def predict(self, predict_x):
        pass


def plot(x, y, w):
    pass


if __name__ == "__main__":
    pass