import numpy as np
import matplotlib.pyplot as plt


class SVM(object):
    """ An easy implementation of SVM algorithm in machine learning.

        A standard SVM algorithm could be represented as:
        
            min     0.5 * (w ^ T * w)
        subject to: train_yi * (w ^ T * train_xi) >= 1

        If support kernel function, then the the programming problem become:
        
            min     0.5 * (w ^ T * w)
        subject to: train_yi * (w ^ T * kernel(train_xi)) >= 1

        If soft margin is supported,
        Consider a slack variable slack_i >= 0, then the programming problem become:
        
            min     0.5 * (w ^ T * w) + C * sum( error_function(slack_i) )
        subject to: train_yi * (w ^ T * train_xi) >= 1 - slack_i
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
        self.parameter = self._train()

    def _sequential_minimal_optimization(self):
        """ SMO is developed by John Platt (Microsoft Research in 1998) to solve Quadratic programming in SVM
        Quadratic Programming could be represented as:
        
            min     sum(a_i) - 0.5 * sum(sum(train_yi * train_yj * kernel(train_xi, train_xj) * a_i * a_j))
        subject to: 0 <= a_i <= C 
                    sum(train_yi * a_i) = 0
                    
        The algorithm proceeds as follows:
            1. Find a Lagrange multiplier that violates KKT conditions.
            2. Pick a second multiplier and optimize the pair.
            3. Repeat steps 1 and 2 until convergence.
        
        And KKT conditions are:
            1. a_i >= 0
            2. a_i * (train_yi * (w ^ T * kernel(train_xi)) - 1) = 0
            3. sum(train_yi * a_i) = 0
        :return: 
        """
        qpmin, qpmin_tmp = 99999999, 99999999
        while qpmin != qpmin_tmp:
            qpmin = qpmin_tmp
            qpmin_tmp = 123


    def _train(self):
        """ Use train_x, train_y to tunes parameter of svm models
        :return: parameter of the model represented by a numpy vector
        """
        return np.zeros(1, 10)

    def test(self, test_x, test_y):
        pass

    def predict(self, predict_x:np.array):
        x = np.concatenate((np.array([1]), predict_x))
        t = x.dot(self.parameter)
        if t > 0:
            return 1
        else:
            return -1


def plot(x, y, w):
    pass


if __name__ == "__main__":
    pass