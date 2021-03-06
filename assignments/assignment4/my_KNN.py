import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
        return

    def dist(self,x):
        # Calculate distances of training data to a single input data point (np.array)


        target = (self.X).to_numpy()

        # distances = {index: distance}
        if self.metric == "minkowski":
            distances = {}
            for i in range(len(target)):
                length = 0
                for k in range(len(x)):
                    length += (abs(x[k]-target[i][k]))**self.p
                distances[i] = length**(1/self.p)


        elif self.metric == "euclidean":
            distances = {}
            for i in range(len(target)):
                length = 0
                for k in range(len(x)):
                    length += (x[k]-target[i][k])**2
                distances[i] = length**(1/2)



        elif self.metric == "manhattan":
            distances = {}
            for i in range(len(target)):
                length = 0
                for k in range(len(x)):
                    length += abs(x[k]-target[i][k])
                distances[i] = length



        elif self.metric == "cosine":
            distances = {}

            # temp_x
            length_x = np.linalg.norm(x)
            temp_x = x/length_x

            #build up the temp_target
            temp_target = {}
            for i in range(len(target)):
                length_target = np.linalg.norm(target[i])
                temp_target[i] = target[i]/length_target

            for i in range(len(target)):
                distances[i] = 1 - (temp_x @ temp_target[i])

        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        distances = self.dist(x)
        sorted_d = sorted(distances.items(),key = lambda x: x[1] , reverse= False)[0:self.n_neighbors]
        output = Counter([self.y[sorted_d[i][0]] for i in range(len(sorted_d))])

        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs
