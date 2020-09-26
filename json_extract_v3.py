import csv
import json
import os

import yaml

from label_extract import load_label

DATA_SET="/home/itu/datadisk/dataset"
TRAINING_DIR="/home/itu/datadisk/dataset/data-for-learning"
DATE="20200629"

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


class WriteToCSV():
    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)

        self.file = open(path, 'a')
        self.csv_write = csv.writer(self.file)

    def init_title(self, all_attributes_key):
        self.csv_write.writerow(all_attributes_key)

    def add_rows(self, all_attributes_value):
        self.csv_write.writerow(all_attributes_value)

    def close(self):
        self.file.close()


def return_obj(data):
    if type(data) is not '__main__.Dict':
        data = dictToObj(data)
    return data


def Return_Attribute_List(data):
    # return list(data.keys())
    return list(data.keys()), list(data.values())


def Return_All_Atributes_p(data, attribute_key, all_attributes_value, all_attributes_key):

    blacklist = ['']
    data = dictToObj(data)
    if isinstance(data, list):
        for item in data:
            # key_ = attribute_key + str(data.index(item))
            if isinstance(item, dict):
                attribute_key_list, attribute_value_list = Return_Attribute_List(item)

                if 'name' in attribute_key_list:
                    key_ = attribute_key + '#' + str(attribute_value_list[attribute_key_list.index('name')])
                else:
                    key_ = attribute_key + str(data.index(item))
                for i in attribute_key_list:
                    item[i] = dictToObj(item[i])
                    Return_All_Atributes_p(item[i], key_ + '/' + str(i), all_attributes_value, all_attributes_key)
            else:
                key_ = attribute_key + str(data.index(item))
                if key_ not in blacklist:
                    if isinstance(data, (int, float)):
                        if (type(data) != bool):
                            all_attributes_value.append(data)
                            all_attributes_key.append(str(attribute_key))
    else:
        if isinstance(data, dict):
            attribute_key_list, attribute_value_list = Return_Attribute_List(data)
            for item in attribute_key_list:
                data[item] = dictToObj(data[item])
                Return_All_Atributes_p(data[item], attribute_key + '/' + str(item), all_attributes_value,
                                     all_attributes_key)
        else:
            if str(attribute_key) not in blacklist:
                if isinstance(data, (int, float)):
                    if (type(data) != bool):
                        all_attributes_value.append(data)
                        all_attributes_key.append(str(attribute_key))

def Return_All_Atributes_v(data, attribute_key, all_attributes_value, all_attributes_key):

    blacklist = ['/devices#IntGW-01/progress', '/devices#IntGW-02/progress', '/devices#RR-01/progress', '/devices#TR-01/progress',
                         '/devices#TR-02/progress', ]
    data = dictToObj(data)
    if isinstance(data, list):
        for item in data:
            # key_ = attribute_key + str(data.index(item))
            if isinstance(item, dict):
                attribute_key_list, attribute_value_list = Return_Attribute_List(item)

                if 'name' in attribute_key_list:
                    key_ = attribute_key + '#' + str(attribute_value_list[attribute_key_list.index('name')])
                else:
                    key_ = attribute_key + str(data.index(item))
                for i in attribute_key_list:
                    item[i] = dictToObj(item[i])
                    Return_All_Atributes_v(item[i], key_ + '/' + str(i), all_attributes_value, all_attributes_key)
            else:
                key_ = attribute_key + str(data.index(item))
                if key_ not in blacklist:
                    if isinstance(data, (int, float)):
                        if (type(data) != bool):
                            all_attributes_value.append(data)
                            all_attributes_key.append(str(attribute_key))
    else:
        if isinstance(data, dict):
            attribute_key_list, attribute_value_list = Return_Attribute_List(data)
            for item in attribute_key_list:
                data[item] = dictToObj(data[item])
                Return_All_Atributes_v(data[item], attribute_key + '/' + str(item), all_attributes_value,
                                     all_attributes_key)
        else:
            if str(attribute_key) not in blacklist:
                if isinstance(data, (int, float)):
                    if (type(data) != bool):
                        all_attributes_value.append(data)
                        all_attributes_key.append(str(attribute_key))

def Return_All_Atributes_n(data, attribute_key, all_attributes_value, all_attributes_key, nexthop, prefix):

    blacklist = ['']
    data = dictToObj(data)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                attribute_key_list, attribute_value_list = Return_Attribute_List(item)

                if attribute_key.split('/')[-1] == 'devices' or \
                        attribute_key.split('/')[-1] == 'neighbor' or attribute_key.split('/')[
                    -1] == 'bgp-neighbor-summary':
                    key_ = attribute_key  # + '#' + str(data.index(item))
                    # print ('ss')
                    # key_ = attribute_key + '#' + str(data.index(item))
                # elif attribute_key.split('/')[-1] == 'neighbor' or attribute_key.split('/')[-1] == 'bgp-neighbor-summary':
                # key_ = attribute_key + '#' + str(data.index(item))
                elif attribute_key.split('/')[-1] == 'bgp-route-entry':
                    if str(attribute_value_list[0]) not in prefix:
                        prefix.append(str(attribute_value_list[0]))
                    key_ = attribute_key  # + '#' + str(attribute_value_list[0])
                elif attribute_key.split('/')[-1] == 'bgp-path-entry':
                    if str(attribute_value_list[0]) not in nexthop:
                        nexthop.append(str(attribute_value_list[0]))
                        # prefix.append(int(1))
                    # else:
                    # prefix[nexthop.index(str(attribute_value_list[0]))] += 1
                    return
                    # key_ = attribute_key + '#' + str(attribute_value_list[0])
                # elif attribute_key.split('/')[-1] == 'load-avg-minute':
                # key_ = attribute_key + '#' + str(attribute_value_list[0])
                # elif 'name' in attribute_key_list:
                # key_ = attribute_key + '#' + str(attribute_value_list[attribute_key_list.index('name')])
                else:
                    key_ = attribute_key
                for i in attribute_key_list:
                    item[i] = dictToObj(item[i])
                    Return_All_Atributes_n(item[i], key_ + '/' + str(i), all_attributes_value, all_attributes_key,
                                           nexthop, prefix)
            else:
                key_ = attribute_key + str(data.index(item))
                if key_ not in blacklist:
                    if isinstance(item, (int, float)):
                        if (type(item) != bool):
                            all_attributes_value.append(item)
                            all_attributes_key.append(key_)
    else:
        if isinstance(data, dict):
            attribute_key_list, attribute_value_list = Return_Attribute_List(data)
            if attribute_key.split('/')[-1] == 'devices' or attribute_key.split('/')[-1] == 'neighbor' \
                    or attribute_key.split('/')[-1] == 'bgp-neighbor-summary':
                key_ = attribute_key
            elif attribute_key.split('/')[-1] == 'bgp-route-entry':
                if str(attribute_value_list[0]) not in prefix:
                    prefix.append(str(attribute_value_list[0]))
                key_ = attribute_key  # + '#' + str(attribute_value_list[0])
            elif attribute_key.split('/')[-1] == 'bgp-path-entry':

                if str(attribute_value_list[0]) not in nexthop:
                    nexthop.append(str(attribute_value_list[0]))
                    # prefix.append(int(1))
                # else:
                # prefix[nexthop.index(str(attribute_value_list[0]))] += 1
                return

                # key_ = attribute_key + '#' + str(attribute_value_list[0])
            # elif attribute_key.split('/')[-1] == 'load-avg-minute':
            # key_ = attribute_key + '#' + str(attribute_value_list[0])
            # elif 'name' in attribute_key_list:
            # key_ = attribute_key + '#' + str(attribute_value_list[attribute_key_list.index('name')])
            else:
                key_ = attribute_key
            for item in attribute_key_list:
                data[item] = dictToObj(data[item])
                Return_All_Atributes_n(data[item], key_ + '/' + str(item), all_attributes_value,
                                       all_attributes_key, nexthop, prefix)
        else:
            if str(attribute_key) not in blacklist:
                if isinstance(data, (int, float)):
                    if (type(data) != bool):
                        all_attributes_value.append(data)
                        all_attributes_key.append(str(attribute_key))

def read_json_by_folder(folder_path, data_type, batch=0, common_file_list=[]):
    path_list = []
    for file_path in os.listdir(folder_path):
        if DATE not in file_path:
            continue
        if len(common_file_list) != 0 and file_path not in common_file_list:
            continue
        if file_path.endswith(".json"):
            path_list.append(file_path)
    path_list.sort(key=lambda x: int(x[:-5]))

    print('file_count:', len(path_list))
    if batch == 0:
        batch = len(path_list)

    # 初始化
    sort_key = []
    new_attributes_value = []
    write_file_path = DATA_SET+'/csv/' + path_list[0][:-11] + data_type+'.csv'
    write_file = None
    recipes = load_label(DATA_SET+"/label-for-learning.json")

    for i in range(batch):
        print(folder_path + path_list[i])
        all_attributes_value, all_attributes_key = read_json_by_path(folder_path + path_list[i], data_type)
        #print(len(all_attributes_key))
        # print('all_attributes_key:', all_attributes_key)
        # print('all_attributes_value', all_attributes_value)
        if i == 0:
            new_key = []
            new_value = []
            for index, element in enumerate(all_attributes_key):
                if element not in new_key:
                    new_key.append(element)
                    new_value.append(all_attributes_value[index])
            all_attributes_key = new_key
            all_attributes_value = new_value
            print(len(all_attributes_key))
            # b = dict(Counter(sort_key))
            # print({key: value for key, value in b.items() if value > 1})  # 重复元素和重复次数
            sort_key = new_key
            #sort_key = all_attributes_key
            write_file = WriteToCSV(write_file_path)

            all_attributes_key.append('type')
            all_attributes_key.append('type_code')
            write_file.init_title(all_attributes_key)

            all_attributes_value.append(recipes.get_type(path_list[i]))
            all_attributes_value.append(recipes.get_type_code(path_list[i]))
            write_file.add_rows(all_attributes_value)
        else:
            new_key = []
            new_value = []
            for index, element in enumerate(all_attributes_key):
                if element not in new_key:
                    new_key.append(element)
                    new_value.append(all_attributes_value[index])
            all_attributes_key = new_key
            all_attributes_value = new_value
            print(len(all_attributes_key))
            new_attributes_value = sort_attributes_value(sort_key, all_attributes_key, all_attributes_value)
            new_attributes_value.append(recipes.get_type(path_list[i]))
            new_attributes_value.append(recipes.get_type_code(path_list[i]))
            write_file.add_rows(new_attributes_value)
        # print('new_attributes_value', new_attributes_value)
    write_file.close()


def sort_attributes_value(sort_key, all_attributes_key, all_attributes_value):
    new_attributes_value = all_attributes_value.copy()

    lack_num = 0
    for i in range(len(sort_key)):
        if sort_key[i] == 'type_code' or sort_key[i] == 'type':
            continue
        if sort_key[i] in all_attributes_key:
            pass
        else:
            lack_num = lack_num + 1
            #print (sort_key[i])
        #new_attributes_value[sort_key.index(all_attributes_key[i])] = all_attributes_value[i]
    #print ("Lack num : ", lack_num)
    for i in range(len(all_attributes_key)):
        # print (all_attributes_key[i])
        new_attributes_value[sort_key.index(all_attributes_key[i])] = all_attributes_value[i]
    return new_attributes_value


def read_json_by_path(path, data_type):
    with open(path, 'r') as load_f:
        data = json.load(load_f)
        data = dictToObj(data)
        attribute_key = ''
        all_attributes_value = []
        all_attributes_key = []
        nexthop = []
        prefix = []
        if data_type == 'v':
            Return_All_Atributes_v(data, attribute_key, all_attributes_value, all_attributes_key)
        elif data_type == 'n':
            Return_All_Atributes_n(data, attribute_key, all_attributes_value, all_attributes_key, nexthop, prefix)
        else:
            Return_All_Atributes_p(data, attribute_key, all_attributes_value, all_attributes_key)

        print(len(nexthop))
        print(len(prefix))
        all_attributes_key.append('nexthop')
        all_attributes_key.append('prefix')
        all_attributes_value.append(len(nexthop))
        all_attributes_value.append(len(prefix))
        return all_attributes_value, all_attributes_key



def main():
    p_path = TRAINING_DIR+'/physical-infrastructure-bgpnw2/'
    n_path = TRAINING_DIR+'/network-device-bgpnw2/'
    v_path = TRAINING_DIR+'/virtual-infrastructure-bgpnw2/'

    p_file_list = os.listdir(p_path)
    n_file_list = os.listdir(n_path)
    v_file_list = os.listdir(v_path)

    common_file_list = [i for i in p_file_list if i in n_file_list if i in v_file_list]

    try:
        read_json_by_folder(p_path, 'p', 0, common_file_list)
    except:
        print("except");
    try:
        read_json_by_folder(n_path, 'n', 0, common_file_list)
    except:
        print("except");
    try:
        read_json_by_folder(v_path, 'v', 0, common_file_list)
    except:
        print("except");

if __name__ == "__main__":
    main()
