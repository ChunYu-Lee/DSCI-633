import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

	def __init__(self, alpha=1):
		# alpha: smoothing factor
		# P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
		# where n(i) is the number of available categories (values) of feature i
		# Setting alpha = 1 is called Laplace smoothing
		self.alpha = alpha

	def fit(self, X, y):
		# turn nan in y into N as a type of category
		if y.isnull().values.any() == True:
			y = y.fillna("N")


		# turn nan into abc as a type of category
		if X.isnull().values.any() == True:
			X = X.fillna("abc")

		# print(X.isnull().values.any())-- can test your nan value
		# print(X.isnull().sum())

		# X: pd.DataFrame, independent variables, str
		# y: list, np.array or pd.Series, dependent variables, int or str
		# list of classes for this model
		self.classes_ = list(set(list(y)))
		# for calculation of P(y)
		self.P_y = Counter(y)
		# self.P[yj][Xi][xi] = P(xi|yi) where Xi is the feature name and xi is the feature value, yj is a specific class label
		self.P = {}

		# classes= list(set(list(y)))

		# target_idx (dict)--> get all the index of every label in classes_.
		target_idx = {}

		for i in self.classes_:
			target_list_idx = []
			for j in range(len(y)):
				if y[j] == i:
					target_list_idx.append(j)
			target_idx[i] = target_list_idx

		#pre-calculate

		for label in self.classes_:
			# independent (dict)-->key:Xi, value: Counter of all the xi under Xi
			# key_dict (dict)-->key:Xi, value: the probability of distinct xi
			independent = {}
			key_dict = {}

			# self.P[label] = {}

			for key in X:
			# Xi_value (dict)--> get total numbers in the specific key and the specific label.
				Xi_value = Counter(X[key][target_idx[label]])
				num = list(set(X[key]))
				for i in range(len(num)):
            		if (num[i] not in Xi_value):
                		Xi_value[num[i]] = 0
				independent[key] = Xi_value

				# prob (dict) --> key:distinct xi, value: the probability of the distinct xi
				prob = {}
				for i in independent[key]:
					# lap = (nc + alpha)/(n + alpha*(# of type of variable))
					lap = (independent[key][i]+ self.alpha)/ (self.P_y[label] + (self.alpha * len(independent[key])) )

					# self.P[label][key][i] = lap
					prob[i] = lap
				key_dict[key] = prob
			self.P[label] = key_dict

		self.P = pd.DataFrame(self.P)

		return

	def predict_proba(self, X):
		# X: pd.DataFrame, independent variables, str
		# prob is a dict of prediction probabilities belonging to each categories
		# return probs = pd.DataFrame(list of prob, columns = self.classes_)
		# write your code below
		# turn nan into abc as a type of category
		if X.isnull().values.any() == True:
			X = X.fillna("abc")

		probs = {}
		for label in self.classes_:
			#[]
			p = self.P_y[label] / len(X)
			for key in X:
				p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
			probs[label] = p
		probs = pd.DataFrame(probs, columns=self.classes_)
		sums = probs.sum(axis=1)
		probs = probs.apply(lambda v: v / sums)
		return probs

	def predict(self, X):
		# X: pd.DataFrame, independent variables, str
		# return predictions: list
		# write your code below
		probs = self.predict_proba(X)
		predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
		return predictions





