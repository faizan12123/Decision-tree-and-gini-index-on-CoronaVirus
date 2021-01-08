import numpy as np
import pandas as pd
import pydotplus
import graphviz
from pandas import DataFrame,Series
from IPython.display import Image
try:
    from STringIO import StringIO # Python 2
except ImportError:
    from io import StringIO # Python 3
import pydotplus
from sklearn import preprocessing
from sklearn import tree
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz 2.44.1\bin'
import timeit
start = timeit.default_timer()


def convert_yes_no(txt):
    if 'yes' in txt:
        return 1
    else:
        return 0


def plot_decision_tree(clf,feature_name,target_name): 
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data,
                         feature_names=feature_name,
                         class_names=target_name,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())

df = pd.read_csv(r"C:\Users\User\.spyder-py3\surveyCSV.csv")
df

df.Scholarship=df.Scholarship.apply(convert_yes_no)
df.Dependent = df.Dependent.map({'dependent': 1, 'independent': 0})
df.Performance=df.Performance.apply(convert_yes_no)
df.Passing=df.Passing.apply(convert_yes_no)
df.Dropped=df.Dropped.apply(convert_yes_no)

df
df=pd.get_dummies(df)
df
X_train=df.loc[:,df.columns != 'Dropped']
Y_train=df.Dropped
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X_train,Y_train)
plot_decision_tree(clf, X_train.columns,df.columns[1])

stop = timeit.default_timer()
print('Time: ', stop - start)  
