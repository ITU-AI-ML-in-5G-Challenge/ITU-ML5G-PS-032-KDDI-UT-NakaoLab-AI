import json
import os

import yaml


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

    if batch == 0:
        batch = len(path_list)

    for i in range(batch):
        print(folder_path + path_list[i])
        read_json_by_path(folder_path + path_list[i])


def read_json_by_path(path):
    with open(path, 'r') as load_f:
        data = json.load(load_f)
        data = dictToObj(data)
        attribute_key = ''
        all_attributes_value = []
        all_attributes_key = []
        Return_All_Atributes(data, attribute_key, all_attributes_value, all_attributes_key)
        print(all_attributes_value)
        print(all_attributes_key)


def main():
    with open("./conf/local_conf.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_list = yaml.load(file, Loader=yaml.FullLoader)
        read_json_by_folder(param_list["physical"], 5)


if __name__ == "__main__":
    main()
