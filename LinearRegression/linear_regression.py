import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    """ An easy implementation of linear regression algorithm in machine learning.

        Form of prediction function is f(x) = w * basic_function(x) is linear in w. So called linear regression
        And this algorithm have a close solution.

        The choice of basic function is part of pre-processing, so when initialize a LinearRegression model,
        X should be pre-processing first, aka. X should be basic_function(origin_x).

        So for a data point x and w, prediction would be w^T * x.
        Then define a error function represents the penalty when our prediction is precisely correct.
        Usually used error function is square errors: errors = ||X * w - Y|| ^ 2 (square of L2 norm)
            Another options is errors = sum(|w^T * xi - yi|)  (L1 norm)

        then this algorithm is:
            min Error_function(w)
        It is a optimization problem without constraints, can get the solution directly by set gradient to zero.

        Take square errors as example, finally, we need to solve w in formula:
            gradient = X^T * Y - (X^T * X) * w = 0
        If X^T * X is a non-singular matrix, we can get w = (X^T * X)^-1 * X^T * Y.
        But usually, it is a singular matrix. There are 2 solutions:
            1, Add regularization to make it non-singular.
            2, Subgradient method

    """

    def __init__(self, x: np.matrix, y: np.ndarray, error_function=None, gradient_function=None):
        """
        Args:
            x: a matrix of train features.
            y: a vertical vector
            error_function: penalty of points within margin or miss classified.
        """
        self.train_x = x
        self.train_y = y
        self.train_data_cnt, self.parameter_cnt = self.train_x.shape
        self.parameter = np.random.random(self.parameter_cnt)
        if error_function is None:
            self.error_function = self._square_error
            self.gradient_function = self._square_error_gradient
            # For gradient in the description, we can compute some part of it before to speed training process.
            self._xy = self.train_x.T.dot(self.train_y)
            self._xtx = self.train_x.T.dot(self.train_x)

    def linear_regression_train(self, max_iteration=1000, epsilon=0.01):
        """ Use train_x, train_y to tunes parameter of svm models
            In this case, use sub-gradient method
        """
        omega = self.parameter
        last_error = self.error_function(self.parameter)
        error = last_error + 1
        alpha, iter_cnt = 2, 0
        while abs(error - last_error) > epsilon and iter_cnt < max_iteration:
            gradient = self.gradient_function(self.parameter)
            error = last_error + 1
            while error >= last_error:
                alpha /= 2.0
                self.parameter = omega - alpha * gradient
                error = self.error_function(self.parameter)
            omega = self.parameter
            last_error = error
            iter_cnt += 1

    def ridge_regression_train(self, alpha):
        """ Use train_x, train_y to tunes parameter of svm models
            In this method, add regularization to avoid singular matrix.
        """
        self.parameter = (self._xtx + alpha * np.identity(self.parameter_cnt)).I.dot(self._xy)

    def predict(self, predict_x: np.matrix):
        return predict_x.dot(self.parameter_cnt)

    def _square_error(self, omega: np.ndarray):
        """ Actually it is not the error function. It is the gradient of the error function over w.
        :return: the function represents gradient of the error function.
        """
        tmp = self.train_x.dot(omega)
        return tmp.T.dot(tmp)

    def _square_error_gradient(self, omega: np.ndarray):
        return self._xy - self._xtx.dot(omega)

