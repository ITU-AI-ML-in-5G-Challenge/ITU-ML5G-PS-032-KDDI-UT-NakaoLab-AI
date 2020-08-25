import csv
import json
import os
import pandas as pd
import time, datetime
import yaml
import matplotlib.pyplot as plt

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
        #if os.path.exists(path):
            #os.remove(path)

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

    attribute_list = ['hardware-cpu-util', 'hardware-disk-size-used', 'hardware-memory-used', 'hardware-network-ip-incoming-datagrams',
                      'hardware-network-ip-outgoing-datagrams', 'hardware-system_stats-io-incoming-blocks',
                      'hardware-system_stats-io-outgoing-blocks',]
    data = dictToObj(data)

    all_attributes_key.append("host_ip")
    for att in attribute_list:
        all_attributes_key.append(att)
    all_attributes_key.append("time")

    value_dict = dictToObj(dictToObj(data.computes[0]).metrics.hardware)
    all_attributes_value.append(dictToObj(data.computes[0]).host_ip)
    for att in attribute_list:
        all_attributes_value.append(value_dict[att])
    all_attributes_value.append(data.time)


def read_json_by_path(path, data_type):
    with open(path, 'r') as load_f:
        data = json.load(load_f)
        data = dictToObj(data)
        attribute_key = ''
        all_attributes_value = []
        all_attributes_key = []
        if data_type == 'v':
            Return_All_Atributes_v(data, attribute_key, all_attributes_value, all_attributes_key)
        elif data_type == 'n':
            Return_All_Atributes_n(data, attribute_key, all_attributes_value, all_attributes_key)
        else:
            Return_All_Atributes_p(data, attribute_key, all_attributes_value, all_attributes_key)
        return all_attributes_value, all_attributes_key

def toTime(timeStamp):
    dateArray = datetime.datetime.fromtimestamp(timeStamp)
    toDate = dateArray.strftime("%Y--%m--%d %H:%M:%S")
    return toDate

def plot(col, time_list, title):
    time_list = pd.to_datetime(time_list)
    plt.scatter(time_list, col)
    plt.title('Changes of ' + title)
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel(title)
    plt.show()
    plt.savefig('./csv_physical/' + title + '.png')

def show_plot(path):
    for file_path in os.listdir(path):
        if file_path.endswith(".csv"):
            f_path = path + file_path
            time_list = []
            len_ = 0
            host_ip = ''
            with open(f_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if i == 0:
                        len_ = len(row)
                        continue
                    else:
                        time_list.append(toTime(int(row[-1])))
                        host_ip = row[0]
            for i in range(1, len_ - 1):
                with open(f_path, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    col1 = [row[i] for row in reader]
                    attribute = col1[0]
                    col_1 = col1[1:]
                    plot (col_1, time_list, attribute + '#' + host_ip)

def read_json_by_folder(folder_path, data_type, batch=0):
    path_list = []
    for file_path in os.listdir(folder_path):
        if file_path.endswith(".json"):
            path_list.append(file_path)
    path_list.sort(key=lambda x: int(x[:-5]))

    print('file_count:', len(path_list))
    if batch == 0:
        batch = len(path_list)

    # 初始化
    #write_file_path = './csv_physical/' + "attributes" + '.csv'
    #write_file = None

    for i in range(batch):
        print(folder_path + path_list[i])
        with open(folder_path + path_list[i], 'r') as load_f:
            data = json.load(load_f)
            data = dictToObj(data)
            host = dictToObj(data.computes[0]).host_ip
            write_file_path = './csv_physical/' + host + '.csv'
        #write_file = None
        all_attributes_value, all_attributes_key = read_json_by_path(folder_path + path_list[i], data_type)
        #print(len(all_attributes_key))
        # print('all_attributes_key:', all_attributes_key)
        # print('all_attributes_value', all_attributes_value)
        if not os.path.exists(write_file_path):
            write_file = WriteToCSV(write_file_path)
            write_file.init_title(all_attributes_key)
            write_file.add_rows(all_attributes_value)
        else:
            write_file = WriteToCSV(write_file_path)
            write_file.add_rows(all_attributes_value)
        write_file.close()



def main():
    with open("./conf/local_conf.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_list = yaml.load(file, Loader=yaml.FullLoader)

        read_json_by_folder(param_list["physical"], 'p', 0)
        #read_json_by_folder(param_list["network"], 'n', 0)
        #read_json_by_folder(param_list["virtual"], 'v', 0)

        path = './csv_physical/'
        show_plot(path)


if __name__ == "__main__":
    main()
