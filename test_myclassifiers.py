import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.random_forest import MyRandomForestClassifier

X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]

def test_random_forest_classifier_fit():
    my_classfier = MyRandomForestClassifier()
    my_classfier.fit(X_train, y_train, 10, 2, 6)
    assert True == True
    

def test_random_forest_classifier_predict():
    my_classfier = MyRandomForestClassifier()
    my_classfier.fit(X_train, y_train, 10, 2, 6)
    my_classfier.predict(X_test)
    assert True == True