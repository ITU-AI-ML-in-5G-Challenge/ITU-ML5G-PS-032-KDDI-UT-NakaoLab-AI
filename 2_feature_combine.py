import pandas as pd
import yaml

with open('./conf.yaml', 'r') as f:
    conf = yaml.load(f.read(), Loader=yaml.SafeLoader)
    train_path = conf['TRAIN_PATH']
    test_path = conf['TEST_PATH']
    dataset_path = conf['CSV_DATA_SET']
    testset_path = conf['CSV_TEST_SET']


# Combining three types of csv.
def df_combine(train_n_files, train_v_files, train_p_files, test_n_files, test_v_files, test_p_files):
    li_n = []
    li_v = []
    li_p = []
    li_test_n = []
    li_test_v = []
    li_test_p = []
    # train
    for filename in train_n_files:
        print('read_csv network:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_n.append(df)
    for filename in train_v_files:
        print('read_csv virtual:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_v.append(df)
    for filename in train_p_files:
        print('read_csv physical:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_p.append(df)
    # test
    for filename in test_n_files:
        print('read csv-for-evaluation network:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_test_n.append(df)
    for filename in test_v_files:
        print('read csv-for-evaluation virtual:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_test_v.append(df)
    for filename in test_p_files:
        print('read csv-for-evaluation physical:', filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        li_test_p.append(df)
    dataset_n = pd.concat(li_n, axis=0, ignore_index=True, sort=False)
    dataset_v = pd.concat(li_v, axis=0, ignore_index=True, sort=False)
    dataset_p = pd.concat(li_p, axis=0, ignore_index=True, sort=False)
    test_n = pd.concat(li_test_n, axis=0, ignore_index=True, sort=False)
    test_v = pd.concat(li_test_v, axis=0, ignore_index=True, sort=False)
    test_p = pd.concat(li_test_p, axis=0, ignore_index=True, sort=False)
    print('dataset_n_v_p:')
    print(dataset_n.shape)
    print(dataset_v.shape)
    print(dataset_p.shape)
    print('testset_n_v_p:')
    print(test_n.shape)
    print(test_v.shape)
    print(test_p.shape)
    dataset_p.drop(['type', 'type_code'], axis=1, inplace=True)
    dataset_n.drop(['type', 'type_code'], axis=1, inplace=True)
    test_p.drop(['type', 'type_code'], axis=1, inplace=True)
    test_n.drop(['type', 'type_code'], axis=1, inplace=True)
    dataset_n.rename(columns=lambda x: 'n_' + x, inplace=True)
    dataset_v.rename(columns=lambda x: 'v_' + x, inplace=True)
    dataset_v['common_time_index'] = dataset_v['v_/time']
    dataset_p.rename(columns=lambda x: 'p_' + x, inplace=True)
    dataset_p['common_time_index'] = dataset_p['p_/time']
    test_n.rename(columns=lambda x: 'n_' + x, inplace=True)
    test_v.rename(columns=lambda x: 'v_' + x, inplace=True)
    test_v['common_time_index'] = test_v['v_/time']
    test_p.rename(columns=lambda x: 'p_' + x, inplace=True)
    test_p['common_time_index'] = test_p['p_/time']
    # dataset = pd.concat([dataset_n, dataset_v, dataset_p], axis=1, sort=False)
    dataset_pn = pd.merge(dataset_p, dataset_n, how='inner', left_index=True, right_index=True)
    dataset = pd.merge(dataset_pn, dataset_v, how='inner', on=['common_time_index'])
    # testset = pd.concat([test_n, test_v, test_p], axis=1, sort=False)
    testset_pn = pd.merge(test_p, test_n, how='inner', left_index=True, right_index=True)
    testset = pd.merge(testset_pn, test_v, how='inner', on=['common_time_index'])
    # Delete the temporary fields used to join the table
    dataset.drop(['common_time_index'], axis=1, inplace=True)
    testset.drop(['common_time_index'], axis=1, inplace=True)
    # Delete lines with outliers
    dataset.dropna(axis=0, how='any', inplace=True)
    testset.dropna(axis=0, how='any', inplace=True)
    return dataset, testset


def main():
    # all_n_files = glob.glob(path + "/*.n.csv")
    # all_v_files = glob.glob(path + "/*.v.csv")
    # all_p_files = glob.glob(path + "/*.p.csv")
    train_n_files = [train_path + x for x in
                     ['20200629.n.csv', '20200630.n.csv', '20200701.n.csv', '20200702.n.csv', '20200703.n.csv',
                      '20200704.n.csv', '20200705.n.csv', '20200706.n.csv']]
    train_v_files = [train_path + x for x in
                     ['20200629.v.csv', '20200630.v.csv', '20200701.v.csv', '20200702.v.csv', '20200703.v.csv',
                      '20200704.v.csv', '20200705.v.csv', '20200706.v.csv']]
    train_p_files = [train_path + x for x in
                     ['20200629.p.csv', '20200630.p.csv', '20200701.p.csv', '20200702.p.csv', '20200703.p.csv',
                      '20200704.p.csv', '20200705.p.csv', '20200706.p.csv']]

    test_n_files = [test_path + x for x in
                    ['20200707.n.csv', '20200708.n.csv', '20200709.n.csv', '20200710.n.csv',
                     '20200711.n.csv', '20200712.n.csv', '20200713.n.csv']]
    test_v_files = [test_path + x for x in
                    ['20200707.v.csv', '20200708.v.csv', '20200709.v.csv', '20200710.v.csv',
                     '20200711.v.csv', '20200712.v.csv', '20200713.v.csv']]
    test_p_files = [test_path + x for x in
                    ['20200707.p.csv', '20200708.p.csv', '20200709.p.csv', '20200710.p.csv',
                     '20200711.p.csv', '20200712.p.csv', '20200713.p.csv']]

    dataset, testset = df_combine(train_n_files, train_v_files, train_p_files, test_n_files, test_v_files, test_p_files)
    print('save to csv..')
    dataset.to_csv(dataset_path)
    testset.to_csv(testset_path)
    print('dataset:')
    print(dataset.shape)
    print('testset:')
    print(testset.shape)


if __name__ == '__main__':
    main()
