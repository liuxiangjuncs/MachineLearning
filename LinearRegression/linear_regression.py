import math
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

    def __init__(self, x: np.matrix, y: np.matrix, error_function=None, gradient_function=None):
        """
        Args:
            x: a N * M matrix of train features.
            y: a N * 1 matrix of train tag.
            error_function: penalty of points within margin or miss classified.
        """
        self.train_x = x
        self.train_y = y
        self.train_data_cnt, self.parameter_cnt = self.train_x.shape
        # print("There are {0} train data points, and each is a vector of {1} dimension".format(self.train_data_cnt,
        #                                                                                       self.parameter_cnt))
        self.parameter = np.matrix(np.random.random(self.parameter_cnt)).T
        # print("Init parameter as {0}".format(self.parameter))
        if error_function is None:
            self.error = self._square_error
            self.gradient = self._square_error_gradient
            # For gradient in the description, we can compute some part of it before to speed training process.
            self._xy = self.train_x.T.dot(self.train_y)
            self._xtx = self.train_x.T.dot(self.train_x)

    def train_linear_regression(self, max_iteration=1000, epsilon=0.01):
        """ Use train_x, train_y to tunes parameter of svm models
            In this case, use sub-gradient method
        """
        last_error, error = 0, self.error()
        alpha, iter_cnt = 0.01, 0
        while abs(error - last_error) > epsilon and iter_cnt < max_iteration:
            last_error = error
            last_para = self.parameter

            gradient = self.gradient()

            self.parameter = last_para - alpha * gradient
            error = self.error()

            # TODO: How to adjust learning rate to make algorithm converge faster?
            # while error >= last_error and alpha != 0:
            #     alpha /= 2.0
            #     print("Change learning rate to {0}".format(alpha))
            #     self.parameter = last_para - alpha * gradient
            #     error = self.error()

            iter_cnt += 1
            print("last_error = {0}, error = {1}".format(last_error, error))

    def train_ridge_regression(self, alpha):
        """ Use train_x, train_y to tunes parameter of svm models
            In this method, add regularization to avoid singular matrix.
        """
        self.parameter = (self._xtx + alpha * np.identity(self.parameter_cnt)).I.dot(self._xy)

    def predict(self, predict_x: np.matrix):
        return predict_x.dot(self.parameter)

    def _square_error(self):
        """ Actually it is not the error function. It is the gradient of the error function over w.
        :return: the function represents gradient of the error function.
        """
        # tmp = self.train_x.dot(self.parameter)
        tmp = self.predict(self.train_x)
        return tmp.T.dot(tmp)

    def _square_error_gradient(self):
        return self._xtx.dot(self.parameter) - self._xy


def generate_data(n: int):
    x, y = [], []
    for i in range(n):
        x.append(np.random.random())
        y.append(2 * math.sin(x[-1] * 2 * math.pi) + np.random.normal(scale=0.5))
    return x, y


def plot_data(x: list, y: list):
    plt.scatter(x, y, color='red', s=10)


def plot_origin():
    x = [1.0 / 100 * _ for _ in range(100)]
    y = [2 * math.sin(_ * 2 * math.pi) for _ in x]
    plt.plot(x, y, color='blue')


def plot_prediction(lr: LinearRegression, degree: int, color: str):
    x = [1.0 / 100 * _ for _ in range(100)]
    test_x = np.matrix([basic_function(_, degree) for _ in x])
    test_y = lr.predict(test_x)
    plt.plot(x, test_y, color=color)


def basic_function(x, k):
    return [x ** _ for _ in range(k + 1)]

if __name__ == "__main__":
    """ the real function is y = 2 * sin(x * 2 * math.pi), 
        And we try to use y = w0 + w1 * x + w2 * x^2 + .... + wn * x^n to predict
    """
    degree = 7

    x, y = generate_data(100)
    train_x = np.matrix([basic_function(_, degree) for _ in x])
    train_y = np.matrix(y).T
    plot_data(x, y)
    plot_origin()

    lr = LinearRegression(train_x, train_y)
    lr.train_linear_regression(max_iteration=1000)
    print("Final error by sub-gradient is {0}".format(lr.error()))
    plot_prediction(lr, degree, "green")

    lr.train_ridge_regression(alpha=0.05)
    print("Final error is ridge regression is {0}".format(lr.error()))
    plot_prediction(lr, degree, "orange")

    plt.show()
