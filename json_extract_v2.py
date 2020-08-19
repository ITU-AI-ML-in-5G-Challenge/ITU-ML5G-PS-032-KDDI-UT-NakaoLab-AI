import csv
import json
import os

import yaml

from label_extract import load_label


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


def Return_All_Atributes(data, attribute_key, all_attributes_value, all_attributes_key):
    data = dictToObj(data)
    if isinstance(data, list):
        for item in data:
            # key_ = attribute_key + str(data.index(item))
            if isinstance(item, dict):
                attribute_key_list, attribute_value_list = Return_Attribute_List(item)
                if 'name' in attribute_key_list:
                    key_ = attribute_key + '#' + attribute_value_list[attribute_key_list.index('name')]
                else:
                    key_ = attribute_key + str(data.index(item))
                for i in attribute_key_list:
                    item[i] = dictToObj(item[i])
                    Return_All_Atributes(item[i], key_ + '/' + i, all_attributes_value, all_attributes_key)
            else:
                if isinstance(data, (int, float)):
                    all_attributes_value.append(item)
                    all_attributes_key.append(attribute_key)
    else:
        if isinstance(data, dict):
            attribute_key_list, attribute_value_list = Return_Attribute_List(data)
            for item in attribute_key_list:
                data[item] = dictToObj(data[item])
                Return_All_Atributes(data[item], attribute_key + '/' + item, all_attributes_value, all_attributes_key)
        else:
            if isinstance(data, (int, float)):
                all_attributes_value.append(data)
                all_attributes_key.append(attribute_key)


def read_json_by_folder(folder_path, batch=0):
    path_list = []
    for file_path in os.listdir(folder_path):
        if file_path.endswith(".json"):
            path_list.append(file_path)
    path_list.sort(key=lambda x: int(x[:-5]))

    print('file_count:', len(path_list))
    if batch == 0:
        batch = len(path_list)

    # 初始化
    sort_key = []
    new_attributes_value = []
    write_file_path = './csv/' + path_list[0][:-11] + '.csv'
    write_file = None
    recipes = load_label("./label-for-learning.json")

    for i in range(batch):
        print(folder_path + path_list[i])
        all_attributes_value, all_attributes_key = read_json_by_path(folder_path + path_list[i])
        print('all_attributes_key:', all_attributes_key)
        print('all_attributes_value', all_attributes_value)
        if i == 0:
            sort_key = all_attributes_key
            write_file = WriteToCSV(write_file_path)

            all_attributes_key.append('type')
            all_attributes_key.append('type_code')
            write_file.init_title(all_attributes_key)

            all_attributes_value.append(recipes.get_type(path_list[i]))
            all_attributes_value.append(recipes.get_type_code(path_list[i]))
            write_file.add_rows(all_attributes_value)
        else:
            new_attributes_value = sort_attributes_value(sort_key, all_attributes_key, all_attributes_value)
            new_attributes_value.append(recipes.get_type(path_list[i]))
            new_attributes_value.append(recipes.get_type_code(path_list[i]))
            write_file.add_rows(new_attributes_value)
        print('new_attributes_value', new_attributes_value)
    write_file.close()


def sort_attributes_value(sort_key, all_attributes_key, all_attributes_value):
    new_attributes_value = all_attributes_value.copy()
    for i in range(len(all_attributes_key)):
        new_attributes_value[sort_key.index(all_attributes_key[i])] = all_attributes_value[i]
    return new_attributes_value


def read_json_by_path(path):
    with open(path, 'r') as load_f:
        data = json.load(load_f)
        data = dictToObj(data)
        attribute_key = ''
        all_attributes_value = []
        all_attributes_key = []
        Return_All_Atributes(data, attribute_key, all_attributes_value, all_attributes_key)
        return all_attributes_value, all_attributes_key


def main():
    with open("./conf/local_conf.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_list = yaml.load(file, Loader=yaml.FullLoader)
        read_json_by_folder(param_list["physical"], 0)


if __name__ == "__main__":
    main()
