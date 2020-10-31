import pandas as pd
import yaml

with open('./conf.yaml', 'r') as f:
    conf = yaml.load(f.read(), Loader=yaml.SafeLoader)
    dataset_path = conf['CSV_DATA_SET']
    testset_path = conf['CSV_TEST_SET']
    diff_dataset_path = conf['CSV_DIFF_DATA_SET']
    diff_testset_path = conf['CSV_DIFF_TEST_SET']


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


def main():
    print('reading dataset...')
    dataset = pd.read_csv(dataset_path)
    testset = pd.read_csv(testset_path)
    print('dataset', dataset.shape)
    print('testset', testset.shape)
    # Segmentation training tests
    column = dataset.columns
    X_train = dataset[column[:-2]]
    X_test = testset[column[:-2]]
    y_train = dataset[column[-1]]
    y_test = testset[column[-1]]

    print('get diff dataset...')
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

    print('writing to csv..')
    diff_dataset.to_csv(diff_dataset_path)
    diff_testset.to_csv(diff_testset_path)
    print('diff_dataset:', diff_dataset.shape)
    print('diff_testset:', diff_testset.shape)


if __name__ == '__main__':
    main()
