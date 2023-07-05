
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria w.r.t relative change of loss
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        n = self.points.shape[0] #doesnt
        d = self.points.shape[1]
        cent = np.empty([self.K, d])

        for i in range(self.K):
            rando = np.random.randint(n)
            cent[i] = self.points[rando]
        
        return cent
        #raise NotImplementedError

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """

        raise NotImplementedError

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        """        

        n = self.points.shape[0]
        num_clusters = np.empty([n])
        distance = pairwise_dist(self.points, self.centers)
        num_clusters = np.argmin(distance, axis = 1)

        return num_clusters
        #raise NotImplementedError

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """

        updated_centers = np.zeros((self.K, self.points.shape[1]), dtype = float)
        for k in range(self.K):
            if np.sum(self.assignments == k) > 0:
                updated_centers[k, :] = np.mean(self.points[self.assignments == k, :], axis = 0)
            else:
                updated_centers[k, :] = self.centers[k, :]

        self.centers = updated_centers
        return self.centers
        #raise NotImplementedError

    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        
        N = self.points.shape[0]
        loss = 0.0 #single float 
        dist = pairwise_dist(self.points, self.centers)
        
        for n in range(N):
            loss += np.square(dist[n][self.assignments[n]])
            
        return loss
        #raise NotImplementedError

    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.1
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        
        for i in range(self.max_iters):
            # Step 1: Assign each point to its closest center
            distances = np.linalg.norm(self.points[:, np.newaxis, :] - self.centers, axis=2)
            self.assignments = np.argmin(distances, axis=1)

            # Step 2: Update the centers to be the mean of the points assigned to each cluster
            new_centers = []
            for j in range(self.K):
                cluster_points = self.points[self.assignments == j]
                if cluster_points.shape[0] == 0:
                    # Step 3: If a cluster has no points, assign a random point as its center
                    new_center = self.points[np.random.choice(self.points.shape[0])]
                    self.centers[j] = new_center
                else:
                    new_center = np.mean(cluster_points, axis=0)
                    new_centers.append(new_center)

            # Step 3 (continued): Replace empty clusters with new centers
            if len(new_centers) > 0:
                self.centers[:len(new_centers)] = np.array(new_centers)

            # Step 4: Calculate the loss and check for convergence
            old_loss = self.loss
            self.loss = np.sum(np.min(distances, axis=1))
            if i > 0:
                loss_diff = np.abs((self.loss - old_loss) / old_loss)
                if loss_diff < self.rel_tol:
                    break

        return self.centers, self.assignments, self.loss
        """
        for i in range(self.max_iters):
            self.assignments = self.update_assignment()
            self.centers = self.update_centers()
            
            for k in range(self.K):
                if np.sum(self.assignments == k) == 0:
                    self.centers[k] = self.points[np.random.choice(self.points.shape[0], 1, replace=False)]
            currentLoss = self.get_loss()
            if i != 0:
                differenceLoss = abs(currentLoss - previousLoss) / ((currentLoss + previousLoss) / 2)
                if differenceLoss < self.rel_tol:
                    break
            previousLoss = currentLoss

        return self.centers, self.assignments, self.loss
        """


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        N = x.shape[0]
        M = y.shape[0]
        point_x = (x * x).sum(axis = 1).reshape((N,1)) * np.ones(shape = (1,M))
        point_y = (y * y).sum(axis = 1) * np.ones(shape = (N,1))
        dist =  np.sqrt(abs(point_x + point_y -2 * x.dot(y.T)))
        
        return dist
        #raise NotImplementedError
