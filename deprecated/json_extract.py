import json
import yaml

path = ""

with open("./conf/local_conf.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    param_list = yaml.load(file, Loader=yaml.FullLoader)
    path = param_list["path"]
    print(param_list["path"])

all_attributes = []


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


def Return_All_Atributes(data):
    data = dictToObj(data)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                attribute_key_list, attribute_value_list = Return_Attribute_List(item)
                for i in attribute_key_list:
                    item[i] = dictToObj(item[i])
                    Return_All_Atributes(item[i])
            else:
                if isinstance(data, (int, float)):
                    all_attributes.append(item)
    else:
        if isinstance(data, dict):
            attribute_key_list, attribute_value_list = Return_Attribute_List(data)
            for item in attribute_key_list:
                data[item] = dictToObj(data[item])
                Return_All_Atributes(data[item])
        else:
            if isinstance(data, (int, float)):
                all_attributes.append(data)
    # print (all_attributes)


def main():
    with open(path, 'r') as load_f:
        data = json.load(load_f)
        data = dictToObj(data)
        Return_All_Atributes(data)
        print(all_attributes)


if __name__ == "__main__":
    main()
