import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        root_mean = np.mean(np.square(pred - label))
        return np.sqrt(root_mean)

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        if len(x.shape) > 1:
            feat = np.empty((x.shape[0], degree + 1, x.shape[1]))
            for i in range(feat.shape[1]):
                feat[:, i, :] = np.power(x, i)
            return feat
        
        else:
            feat = np.empty((x.shape[0], degree + 1))
            for i in range(feat.shape[1]):
                feat[:, i] = np.power(x, i)
            return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,D) numpy array, where N is the number
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        return np.matmul(xtest, weight)


    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        xtest = np.linalg.pinv(xtrain)
        return self.predict(xtest, ytrain)

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weight = np.zeros((xtrain.shape[1], 1))
        loss_per_epoch = list()

        for x in range(epochs):
            weight += (np.dot(xtrain.T, ytrain - (np.dot(xtrain,weight))) * learning_rate) / ytrain.shape[0]
            
            predict = self.predict(xtrain, weight)
            rmse = self.rmse(predict, ytrain)
            loss_per_epoch.append(rmse)
    
        return weight, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float, value of regularization constant
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        weight = np.zeros((xtrain.shape[1], 1))
        loss_per_step = list()
        for i in range(epochs):
            for y in range(ytrain.shape[0]):
                weight += (learning_rate*(xtrain[y]*(ytrain[y] - (np.dot(xtrain[y],weight) )))).reshape(weight.shape)
                
                predict = self.predict(xtrain, weight)
                rmse = self.rmse(predict, ytrain)
                loss_per_step.append(rmse)
    
        return weight, loss_per_step

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        bias = np.eye(xtrain.shape[1])
        bias[0][0] = 0
        fir_inner = self.predict(xtrain.T, xtrain)
        sec_inner = np.linalg.pinv(fir_inner + c_lambda * bias)
        return self.predict(self.predict(sec_inner, xtrain.T), ytrain)

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-7,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weight = np.zeros((xtrain.shape[1], 1))
        loss_per_epoch = list()
        for i in range(epochs):
            inside = np.dot(xtrain,weight)
            weight += ((np.dot(xtrain.T, ytrain - inside) - (weight * c_lambda)) * learning_rate) / ytrain.shape[0]
            
            pred = self.predict(xtrain, weight)
            rmse = self.rmse(pred, ytrain)
            loss_per_epoch.append(rmse)
        return weight, loss_per_epoch

    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        weight = np.zeros((xtrain.shape[1], 1))
        loss_per_step = list()
        for i in range(epochs):
            for j in range(ytrain.shape[0]):
                inside = np.dot(xtrain[j],weight)
                weight += learning_rate*(((xtrain[j,:].T).reshape(weight.shape))*(ytrain[j] - inside) - (c_lambda * weight)/xtrain.shape[0])
                pred = self.predict(xtrain, weight)
                rmse = self.rmse(pred, ytrain)
                loss_per_step.append(rmse)
    
        return weight, loss_per_step

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> List[float]:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each kfold

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
    
        loss_per_fold = list()
        length = int(X.shape[0] / kfold)
        for i in range(kfold):
            x1 = X[:i * length, :]
            x2 = X[(i + 1) * length:, :]
            x_train = np.concatenate((x1, x2))

            y1 = y[:length*i]
            y2 = y[length*i + length :]
            y_train = np.concatenate((y1, y2))

            x_test = X[length * i:(length * i + length), :]
            y_test = y[length * i:(length * i + length)]

            weights = self.ridge_fit_closed(x_train, y_train, c_lambda)
            prediction = self.predict(x_test, weights)
            loss_per_fold.append(self.rmse(prediction, y_test))
        return loss_per_fold
        

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS
        
        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """
        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm

        return best_lambda, best_error, error_list

