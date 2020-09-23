import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance
import numpy as np
from numpy import sort

# XGBoost
def xgboost(X_train, y_train, X_test, y_test):
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
    plot_confusion_matrix(cm, classes=['0', '1', '2'], normalize=True, title='Normalized confusion matrix')
    plt.show()


if __name__ == '__main__':
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)

    # 读取csv
    path = r'/home/itu/datadisk/dataset/csv-for-learning/'
    # path = r'/Users/xiafei/Downloads/csv-for-learning/'
    # all_n_files = glob.glob(path + "/*.n.csv")
    # all_v_files = glob.glob(path + "/*.v.csv")
    # all_p_files = glob.glob(path + "/*.p.csv")
    all_n_files = [path + x for x in
                   ['20200629.n.csv', '20200630.n.csv', '20200701.n.csv', '20200702.n.csv', '20200703.n.csv',
                    '20200704.n.csv', '20200705.n.csv', '20200706.n.csv']]
    all_v_files = [path + x for x in
                   ['20200629.v.csv', '20200630.v.csv', '20200701.v.csv', '20200702.v.csv', '20200703.v.csv',
                    '20200704.v.csv', '20200705.v.csv', '20200706.v.csv']]
    all_p_files = [path + x for x in
                   ['20200629.p.csv', '20200630.p.csv', '20200701.p.csv', '20200702.p.csv', '20200703.p.csv',
                    '20200704.p.csv', '20200705.p.csv', '20200706.p.csv']]

    li_n = []
    li_v = []
    li_p = []

    for filename in all_n_files:
        print('read_csv network:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_n.append(df)

    for filename in all_v_files:
        print('read_csv virtual:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_v.append(df)

    for filename in all_p_files:
        print('read_csv physical:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_p.append(df)

    dataset_n = pd.concat(li_n, axis=0, ignore_index=True, sort=False)
    dataset_v = pd.concat(li_v, axis=0, ignore_index=True, sort=False)
    dataset_p = pd.concat(li_p, axis=0, ignore_index=True, sort=False)

    print(dataset_n.shape)
    print(dataset_v.shape)
    print(dataset_p.shape)

    dataset_n.drop(['type', 'type_code'], axis=1, inplace=True)
    dataset_v.drop(['type', 'type_code'], axis=1, inplace=True)

    dataset_n.rename(columns=lambda x: 'n_' + x, inplace=True)
    dataset_v.rename(columns=lambda x: 'v_' + x, inplace=True)
    dataset_p.rename(columns=lambda x: 'p_' + x, inplace=True)

    dataset = pd.concat([dataset_n, dataset_v, dataset_p], axis=1, sort=False)
    # 数据集概览
    # print(dataset.describe())
    # print(dataset.head(5))

    # valid
    # print('isnan', np.isnan(dataset.any()))
    # print('isfinite', np.isfinite(dataset.all()))
    # dataset = pd.read_csv('/home/itu/datadisk/dataset/csv-for-learning/20200629.n.csv')

    print('列数:', dataset.shape[1], '行数:', dataset.shape[0])

    column = dataset.columns
    X = dataset[column[:-2]]
    # X= X.values
    Y = dataset[column[-1]]
    # Y= Y.values
    # 划分训练测试
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ss = StandardScaler()
    std_X_train = ss.fit_transform(X_train)
    std_X_test = ss.fit_transform(X_test)

    # 重要度
    # fit model no training data
    model = XGBClassifier(importance_type='gain')
    model.fit(X, Y)

    # plot feature importance
    _, ax = plt.subplots(figsize=(20, 100))
    # * "weight" is the number of times a feature appears in a tree
    # * "gain" is the average gain of splits which use the feature
    # * "cover" is the average coverage of splits which use the feature
    # where coverage is defined as the number of samples affected by the split
    plot_importance(model, ax=ax, max_num_features=200, importance_type='gain')
    plt.show()

    # 结果写入文件
    im = pd.DataFrame({'importance': model.feature_importances_, 'var': dataset.columns[:-2]})
    im = im.sort_values(by='importance', ascending=False)
    im.to_csv("feature_important_data_XG.csv")

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Fit model using each importance as a threshold
    thresholds = sort(model.feature_importances_)[-100:]

    print(thresholds)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))





