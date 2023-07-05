import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    
    #raise NotImplementedError
	additional_features = []
    for i in range(len(X)): 
    	additional_features.append([X[i][1] * X[i][0]])
    return np.hstack((X, additional_features))
