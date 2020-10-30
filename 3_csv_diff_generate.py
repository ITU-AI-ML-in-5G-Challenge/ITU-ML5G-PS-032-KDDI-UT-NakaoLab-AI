import pandas as pd


def get_diff_dataset(dataset, X_train):
    diff_dataset = pd.DataFrame(columns=dataset.columns)
    cur_type = 0
    start_index = 0

    for index, row in dataset.iterrows():
        row_type = row[-1]

        if index == dataset.shape[0] - 1:
            # last line
            end_index = index
            diff_df = X_train.loc[[start_index, end_index]].diff().loc[[end_index]]
            diff_df['v_type_code'] = cur_type
            diff_dataset = diff_dataset.append(diff_df, ignore_index=True, sort=False)

        elif row_type != cur_type:
            end_index = index - 1
            diff_df = X_train.loc[[start_index, end_index]].diff().loc[[end_index]]
            diff_df['v_type_code'] = cur_type
            diff_dataset = diff_dataset.append(diff_df, ignore_index=True, sort=False)
            # new start
            start_index = index
            cur_type = row_type

    diff_dataset['v_type_code'] = pd.to_numeric(diff_dataset['v_type_code'])
    return diff_dataset


if __name__ == '__main__':
    print('reading dataset...')
    dataset = pd.read_csv('./csv/dataset.csv')
    testset = pd.read_csv('./csv/testset.csv')
    print('dataset', dataset.shape)
    print('testset', testset.shape)
    # 划分训练测试
    column = dataset.columns
    X_train = dataset[column[:-2]]
    X_test = testset[column[:-2]]
    y_train = dataset[column[-1]]
    y_test = testset[column[-1]]

    print('获取差值训练集...')
    diff_dataset = get_diff_dataset(dataset, X_train)
    diff_testset = get_diff_dataset(testset, X_test)

    # Remove the useless fields.
    diff_dataset.drop(['v_type'], axis=1, inplace=True)
    diff_testset.drop(['v_type'], axis=1, inplace=True)
    # Remove the effect of time on results.
    diff_dataset.drop(['p_/time'], axis=1, inplace=True)
    diff_testset.drop(['p_/time'], axis=1, inplace=True)
    diff_dataset.drop(['v_/time'], axis=1, inplace=True)
    diff_testset.drop(['v_/time'], axis=1, inplace=True)
    # Delete the first two unknown columns (doesn't work~)
    # diff_dataset.drop(columns=['Unnamed: 0'], inplace=True)
    # diff_testset.drop(columns=['Unnamed: 0'], inplace=True)

    print('写入到csv..')
    diff_dataset.to_csv('./csv/diff_dataset.csv')
    diff_testset.to_csv('./csv/diff_testset.csv')
    print('diff_dataset:', diff_dataset.shape)
    print('diff_testset:', diff_testset.shape)
