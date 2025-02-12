import time
import yaml
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

with open('./conf.yaml', 'r') as f:
    conf = yaml.load(f.read(), Loader=yaml.SafeLoader)
    dataset_path = conf['CSV_DIFF_DATA_SET']
    testset_path = conf['CSV_DIFF_TEST_SET']


# Decision tree

def decision_tree(X_train, y_train, X_test, y_test, show=False):
    last_time = time.time()
    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

    dt.fit(X_train, y_train)
    middle_time = time.time()

    y_pred = dt.predict(X_test)

    current_time = time.time()

    print("DT Accuracy: %.2f" % accuracy_score(y_test, y_pred))

    print("train time： {}".format(middle_time - last_time))
    print("test time： {}".format(current_time - middle_time))
    if show:
        cm = confusion_matrix(y_test, y_pred)
        print('confusion matrix dt:')
        print(cm)
        print('classification report dt:')
        print(classification_report(y_test, y_pred))


# random forest
def random_forest(X_train, y_train, X_test, y_test, show=False, estimators=100):
    last_time = time.time()
    rf = RandomForestClassifier(n_estimators=estimators, max_depth=None, min_samples_split=2, random_state=0)

    rf.fit(X_train, y_train)
    middle_time = time.time()

    y_pred = rf.predict(X_test)

    current_time = time.time()
    print("n_estimators: %d" % estimators)
    print("RF Accuracy: %.2f" % accuracy_score(y_test, y_pred))
    print("train time： {}".format(middle_time - last_time))
    print("test time： {}".format(current_time - middle_time))
    if show:
        cm = confusion_matrix(y_test, y_pred)
        print('confusion matrix rf:')
        print(cm)
        print('classification report rf:')
        print(classification_report(y_test, y_pred))


# XGBoost
def xgboost(X_train, y_train, X_test, y_test, show=False, estimators=100):
    last_time = time.time()
    xgb = XGBClassifier(n_estimators=estimators, n_jobs=-1, objective='multi:softmax', num_class=5, random_state=0)

    xgb.fit(X_train, y_train)
    middle_time = time.time()

    y_pred = xgb.predict(X_test)

    current_time = time.time()

    print("n_estimators: %d" % estimators)
    print("XGBOOST Accuracy: %.2f" % accuracy_score(y_test, y_pred))
    print("train time： {}".format(middle_time - last_time))
    print("test time： {}".format(current_time - middle_time))
    if show:
        cm = confusion_matrix(y_test, y_pred)
        print('confusion matrix xgb:')
        print(cm)
        print('classification report xgb:')
        print(classification_report(y_test, y_pred))


# MLP
def mlp(std_X_train, y_train, std_X_test, y_test, show=False):
    last_time = time.time()
    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(200, 200, 200, 200),
                        random_state=1, max_iter=10000, verbose=0, learning_rate_init=.1)
    mlp.fit(std_X_train, y_train)
    middle_time = time.time()
    y_pred = mlp.predict(std_X_test)
    current_time = time.time()
    print("MLP Accuracy: %.2f" % accuracy_score(y_test, y_pred))
    print("train time： {}".format(middle_time - last_time))
    print("test time： {}".format(current_time - middle_time))
    if show:
        cm = confusion_matrix(y_test, y_pred)
        print('confusion matrix MLP:')
        print(cm)
        print('classification report MLP:')
        print(classification_report(y_test, y_pred))


# SVM
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


def train_svm(std_X_train, y_train, std_X_test, y_test, show=False):
    last_time = time.time()
    model = svm_cross_validation(std_X_train, y_train.ravel())
    middle_time = time.time()
    y_pred = model.predict(std_X_test)
    print("SVM Accuracy: %.2f" % accuracy_score(y_test, y_pred))
    current_time = time.time()
    print("train time： {}".format(middle_time - last_time))
    if show:
        print("test time： {}".format(current_time - middle_time))
        cm = confusion_matrix(y_test, y_pred)
        print('confusion matrix svm:')
        print(cm)
        print('classification report svm:')
        print(classification_report(y_test, y_pred))


def main():
    print('reading diff dataset:')
    dataset = pd.read_csv(dataset_path, index_col=None, header=0)
    testset = pd.read_csv(testset_path, index_col=None, header=0)

    # Delete the following states
    # 0: ixnetwork-traffic-start
    # 2: node-up
    # 4: interface-up
    # 6: tap-loss-stop
    # 8: tap-delay-stop
    # 10: ixnetwork-bgp-injection-stop
    # 12: ixnetwork-bgp-hijacking-stop
    data_drop_index = dataset[dataset['v_type_code'] % 2 == 0].index.tolist()
    dataset.drop(index=data_drop_index, axis=0, inplace=True)
    test_drop_index = testset[testset['v_type_code'] % 2 == 0].index.tolist()
    testset.drop(index=test_drop_index, axis=0, inplace=True)

    # Try to combine 5 and 7 into one type.
    dataset['v_type_code'] = dataset['v_type_code'].apply(lambda x: 57 if x == 5 or x == 7 else x)
    testset['v_type_code'] = testset['v_type_code'].apply(lambda x: 57 if x == 5 or x == 7 else x)

    print('dataset:')
    print(dataset.shape)
    print('testset:')
    print(testset.shape)

    # train test split
    column = dataset.columns
    X_train = dataset[column[:-1]]
    X_test = testset[column[:-1]]
    y_train = dataset[column[-1]]
    y_test = testset[column[-1]]

    X = pd.concat([X_train, X_test], axis=0, ignore_index=True, sort=False)
    Y = pd.concat([y_train, y_test], axis=0, ignore_index=True, sort=False)

    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print(X.shape, Y.shape)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ss = StandardScaler()
    std_X_train = ss.fit_transform(X_train)
    std_X_test = ss.fit_transform(X_test)

    # estimator_list = [100, 300, 500]

    decision_tree(X_train, y_train, X_test, y_test, show=True)

    # for n in estimator_list:
    random_forest(X_train, y_train, X_test, y_test, show=True)
    xgboost(X_train, y_train, X_test, y_test, show=True)

    mlp(std_X_train, y_train, std_X_test, y_test, show=True)

    train_svm(std_X_train, y_train, std_X_test, y_test, show=True)


if __name__ == '__main__':
    main()
