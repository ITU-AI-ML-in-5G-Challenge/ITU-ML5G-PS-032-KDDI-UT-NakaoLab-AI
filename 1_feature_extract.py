import csv
import json
import os
import yaml
from label_extract import load_label

with open('./conf.yaml', 'r') as f:
    conf = yaml.load(f.read(), Loader=yaml.SafeLoader)
    DATA_SET = conf['DATA_SET']
    LEARNING_DIR = conf['LEARNING_DIR']
    EVALUATION_DIR = conf['EVALUATION_DIR']
    PHYSICAL_SUB_DIR = conf['PHYSICAL_SUB_DIR']
    NETWORK_SUB_DIR = conf['NETWORK_SUB_DIR']
    VIRTUAL_SUB_DIR = conf['VIRTUAL_SUB_DIR']
DATE = "20200629"


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
    blacklist = ['/devices#IntGW-01/progress', '/devices#IntGW-02/progress', '/devices#RR-01/progress',
                 '/devices#TR-01/progress',
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
                    key_ = attribute_key
                elif attribute_key.split('/')[-1] == 'bgp-route-entry':
                    if str(attribute_value_list[0]) not in prefix:
                        prefix.append(str(attribute_value_list[0]))
                    key_ = attribute_key  # + '#' + str(attribute_value_list[0])
                elif attribute_key.split('/')[-1] == 'bgp-path-entry':
                    if str(attribute_value_list[0]) not in nexthop:
                        nexthop.append(str(attribute_value_list[0]))
                    return
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
                return
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


def read_json_by_folder(folder_path, data_type, batch, attribute, common_file_list=[]):
    path_list = []
    for file_path in os.listdir(folder_path):
        if len(common_file_list) != 0 and file_path not in common_file_list:
            continue
        if file_path.endswith(".json"):
            path_list.append(file_path)
    path_list.sort(key=lambda x: int(x[:-5]))

    if batch == 0:
        batch = len(path_list)

    # initialization
    sort_key = []
    write_file = None
    recipes = load_label(DATA_SET + '/label-for-' + attribute + '.json')
    cur_date = 'inital_date';
    for i in range(batch):
        write_file_path = DATA_SET + '/csv-for-' + attribute + '/' + path_list[i][:-11] + '.' + data_type + '.csv'
        print(folder_path + path_list[i])
        all_attributes_value, all_attributes_key = read_json_by_path(folder_path + path_list[i], data_type)
        print(cur_date)
        if cur_date not in path_list[i]:
            cur_date = path_list[i][:8];

            new_key = []
            new_value = []
            for index, element in enumerate(all_attributes_key):
                if element not in new_key:
                    new_key.append(element)
                    new_value.append(all_attributes_value[index])
            all_attributes_key = new_key
            all_attributes_value = new_value
            sort_key = new_key
            if write_file:
                write_file.close();
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

            new_attributes_value = sort_attributes_value(sort_key, all_attributes_key, all_attributes_value)
            new_attributes_value.append(recipes.get_type(path_list[i]))
            new_attributes_value.append(recipes.get_type_code(path_list[i]))
            write_file.add_rows(new_attributes_value)
    if write_file:
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
            print(sort_key[i])
    for i in range(len(all_attributes_key)):
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
            all_attributes_key.append('nexthop')
            all_attributes_key.append('prefix')
            all_attributes_value.append(len(nexthop))
            all_attributes_value.append(len(prefix))
        else:
            Return_All_Atributes_p(data, attribute_key, all_attributes_value, all_attributes_key)

        return all_attributes_value, all_attributes_key


def main():
    l_p_path = LEARNING_DIR + PHYSICAL_SUB_DIR
    l_n_path = LEARNING_DIR + NETWORK_SUB_DIR
    l_v_path = LEARNING_DIR + VIRTUAL_SUB_DIR

    e_p_path = EVALUATION_DIR + PHYSICAL_SUB_DIR
    e_n_path = EVALUATION_DIR + NETWORK_SUB_DIR
    e_v_path = EVALUATION_DIR + VIRTUAL_SUB_DIR

    common_file_list = [i for i in os.listdir(l_p_path) if i in os.listdir(l_n_path) if i in os.listdir(l_v_path)]
    test_common_file_list = [i for i in os.listdir(e_p_path) if i in os.listdir(e_n_path) if i in os.listdir(e_v_path)]

    read_json_by_folder(l_p_path, 'p', 0, 'learning', common_file_list)
    read_json_by_folder(l_n_path, 'n', 0, 'learning', common_file_list)
    read_json_by_folder(l_v_path, 'v', 0, 'learning', common_file_list)

    read_json_by_folder(e_p_path, 'p', 0, 'evaluation', test_common_file_list)
    read_json_by_folder(e_n_path, 'n', 0, 'evaluation', test_common_file_list)
    read_json_by_folder(e_v_path, 'v', 0, 'evaluation', test_common_file_list)


if __name__ == "__main__":
    main()
