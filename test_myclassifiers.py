import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myclassifiers import MyRandomForestClassifier

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
    trees = [['Attribute', 'att2', 
                ['Value', 'no', 
                    ['Attribute', 'att0', 
                        ['Value', 'Junior', 
                            ['Attribute', 'att3', 
                                ['Value', 'no', 
                                    ['Leaf', 'True', 1, 2]
                                ], 
                                ['Value', 'yes', 
                                    ['Leaf', 'False', 1, 2]
                                ]
                            ]
                        ], 
                        ['Value', 'Mid', 
                            ['Leaf', 'True', 1, 6]
                        ], 
                        ['Value', 'Senior', 
                            ['Leaf', 'False', 3, 6]
                        ]
                    ]
                ], 
                ['Value', 'yes', 
                    ['Leaf', 'True', 3, 9]
                ]
            ], 
            ['Attribute', 'att0', 
                ['Value', 'Junior', 
                    ['Attribute', 'att3', 
                        ['Value', 'no', 
                            ['Leaf', 'True', 2, 4]
                        ], 
                        ['Value', 'yes', 
                            ['Leaf', 'False', 2, 4]
                        ]
                    ]
                ], 
                ['Value', 'Mid', 
                    ['Leaf', 'True', 4, 9]
                ], 
                ['Value', 'Senior', 
                    ['Leaf', 'False', 1, 9]
                ]
            ], 
            ['Attribute', 'att2', 
                ['Value', 'no', 
                    ['Attribute', 'att0', 
                        ['Value', 'Junior', 
                            ['Attribute', 'att3', 
                                ['Value', 'no', 
                                    ['Leaf', 'True', 1, 2]
                                ], 
                                ['Value', 'yes', 
                                    ['Leaf', 'False', 1, 2]
                                ]
                            ]
                        ], 
                        ['Value', 'Senior', 
                            ['Leaf', 'False', 6, 8]
                        ]
                    ]
                ], 
                ['Value', 'yes', 
                    ['Leaf', 'True', 1, 9]
                ]
            ], 
            ['Attribute', 'att2', 
                ['Value', 'no', 
                    ['Leaf', 'False', 3, 9]
                ], 
                ['Value', 'yes', 
                    ['Leaf', 'True', 6, 9]
                ]
            ], 
            ['Attribute', 'att2', 
                ['Value', 'no', 
                    ['Leaf', 'False', 4, 9]
                ], 
                ['Value', 'yes', 
                    ['Leaf', 'True', 5, 9]
                ]
            ], 
            ['Attribute', 'att0', 
                ['Value', 'Junior', 
                    ['Leaf', 'False', 4, 9]
                ], 
                ['Value', 'Mid', 
                    ['Leaf', 'True', 4, 9]
                ], 
                ['Value', 'Senior', 
                    ['Leaf', 'False', 1, 9]
                ]
            ]]
    assert my_classfier.trees == trees
    

def test_random_forest_classifier_predict():
    my_classfier = MyRandomForestClassifier()
    my_classfier.fit(X_train, y_train, 10, 2, 6)
    y_predicted = my_classfier.predict(X_test)
    y_correct = ["True", "True"]
    assert y_predicted == y_correct