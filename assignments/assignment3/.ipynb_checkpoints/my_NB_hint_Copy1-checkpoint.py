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

		all_possible_value = {key: set(X[key]) for key in X}
		for label in self.classes_:
			# build up the self.P[label]--> empty
			self.P[label] = {}
			filtered = X[y == label]

			for key in X:
			# build up the self.P[label][key] -->empty
				self.P[label][key] = {}
				value = Counter(filtered[key])

				for i in all_possible_value[key]:
					# lap = (nc + alpha)/(n + alpha*(# of type of variable))
					self.P[label][key][i] = (value[i]+ self.alpha)/ (self.P_y[label] + (self.alpha * len(all_possible_value[key])) )

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





