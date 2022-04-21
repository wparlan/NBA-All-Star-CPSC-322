"""mysimplelinearregressor.py
@author gsprint23

Note: is used for the regressor in MySimpleLinearRegressionClassifier
"""
import numpy as np

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, x_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        x_train = [x[0] for x in x_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(x_train,
            y_train)

    def predict(self, x_test):
        """Makes predictions for test samples in x_test.

        Args:
            x_test(list of list of numeric vals): The list of testing samples
                The shape of x_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to x_test)
        """
        predictions = []
        if self.slope is not None and self.intercept is not None:
            for test_instance in x_test:
                predictions.append(self.slope * test_instance[0] + self.intercept)
        return predictions

    @staticmethod # decorator to denote this is a static (class-level) method
    def compute_slope_intercept(x_list, y_list):
        """Fits a simple univariate line y = mx + b to the provided x y data.
        Follows the least squares approach for simple linear regression.

        Args:
            x(list of numeric vals): The list of x values
            y(list of numeric vals): The list of y values

        Returns:
            m(float): The slope of the line fit to x and y
            b(float): The intercept of the line fit to x and y
        """
        mean_x = np.mean(x_list)
        mean_y = np.mean(y_list)
        m_value = sum([(x_list[i] - mean_x) * (y_list[i] - mean_y) for i in range(len(x_list))]) \
            / sum([(x_list[i] - mean_x) ** 2 for i in range(len(x_list))])
        # y = mx + b => y - mx
        b_value = mean_y - m_value * mean_x
        return m_value, b_value
