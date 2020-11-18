import pandas as pd
import random as random
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
import time
import sys

sys.path.insert(0, '..')
from assignment8.my_evaluation import my_evaluation


class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        df = pd.concat([X, y], axis=1)
        df_majority = df[df.fraudulent==0]
        df_minority = df[df.fraudulent==1]
        df_minority_oversampled = resample(df_minority, replace=True,n_samples=len(df_majority),random_state=1234)
        df_oversampled = pd.concat([df_minority_oversampled, df_majority])
        df_oversampled = df_oversampled.sample(frac=1)
        df_x = df_oversampled.drop(['fraudulent'], axis=1)
        df_y = df_oversampled["fraudulent"]
        
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        XX = self.preprocessor.fit_transform(df_x["description"]+df_x["requirements"])
        
        parameters = {'loss':('epsilon_insensitive', 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron','squared_loss', 'huber', 'squared_epsilon_insensitive' ), 'class_weight': ('balanced', 'weight'),'alpha': (1e-1, 1e-4)}
        self.clf = SGDClassifier( loss='epsilon_insensitive',class_weight="balanced", alpha = 0.0001)
        self.gs_clf = GridSearchCV(self.clf, parameters, cv=5, n_jobs=-1).fit(XX, df_y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        XX = self.preprocessor.transform(X["description"]+X["requirements"]) #X["description"] X["requirements"]
        predictions = self.gs_clf.predict(XX)
        return predictions


def test(data):
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    split_point = int(0.8 * len(y))
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    eval = my_evaluation(predictions, y_test)
    f1 = eval.f1(target=1)
    return f1


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("F1 score: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print(runtime)
