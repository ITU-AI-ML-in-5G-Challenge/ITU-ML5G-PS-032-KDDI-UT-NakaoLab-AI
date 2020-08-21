import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red")
        # color="red" if cm[i, j] > thresh else "black")
    # plt.set_tight_layout(True)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 读取csv
dataset = pd.read_csv('./csv/20200629.csv')

print('列数:', dataset.shape[1], '行数:', dataset.shape[0])
# 划分训练测试
column = dataset.columns
X = dataset[column[:-2]]
# X= X.values
Y = dataset[column[-1]]
# Y= Y.values

X_all, y_all = X, Y
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
ss = StandardScaler()
std_X_train = ss.fit_transform(X_train)
std_X_test = ss.fit_transform(X_test)

###########################  ----decision tree

last_time = time.time()
dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

dt.fit(X_train, y_train)
middle_time = time.time()

y_pred = dt.predict(X_test)

current_time = time.time()

accuracy_score(y_test, y_pred)

print("训练耗时： {}".format(middle_time - last_time))
print("测试耗时： {}".format(current_time - middle_time))

cm = confusion_matrix(y_test, y_pred)
print('confusion matrix dt:')
print(cm)
print('classification report dt:')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                      normalize=True, title='Normalized confusion matrix')

plt.show()

###########################  ----random forest
last_time = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)

rf.fit(X_train, y_train)
middle_time = time.time()

y_pred = rf.predict(X_test)

current_time = time.time()

accuracy_score(y_test, y_pred)

print("训练耗时： {}".format(middle_time - last_time))
print("测试耗时： {}".format(current_time - middle_time))
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix rf:')
print(cm)
print('classification report rf:')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                      normalize=True, title='Normalized confusion matrix')

plt.show()

###############################  ----XGBoost
last_time = time.time()
xgb = XGBClassifier(n_estimators=300, objective='multi:softmax', num_class=13, random_state=0)

xgb.fit(X_train, y_train)
middle_time = time.time()

y_pred = xgb.predict(X_test)

current_time = time.time()

accuracy_score(y_test, y_pred)

print("训练耗时： {}".format(middle_time - last_time))
print("测试耗时： {}".format(current_time - middle_time))
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix xgb:')
print(cm)
print('classification report xgb:')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                      normalize=True, title='Normalized confusion matrix')

plt.show()

###############################  ----MLP
last_time = time.time()
mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(200, 200, 200, 200),
                    random_state=1, max_iter=10000, verbose=0, learning_rate_init=.1)
mlp.fit(std_X_train, y_train)
middle_time = time.time()
y_pred = mlp.predict(std_X_test)
current_time = time.time()
print("训练耗时： {}".format(middle_time - last_time))
print("测试耗时： {}".format(current_time - middle_time))
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix MLP:')
print(cm)
print('classification report MLP:')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                      normalize=True, title='Normalized confusion matrix')

plt.show()


###############################  ----SVM

def svm_cross_validation(X, y):
    model = svm.SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(X, y)
    return model


last_time = time.time()
model = svm_cross_validation(std_X_train, y_train.ravel())
middle_time = time.time()
y_pred = model.predict(std_X_test)
current_time = time.time()
print("训练耗时： {}".format(middle_time - last_time))
print("测试耗时： {}".format(current_time - middle_time))
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix svm:')
print(cm)
print('classification report svm:')
print(classification_report(y_test, y_pred))

plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                      normalize=True, title='Normalized confusion matrix')

plt.show()
