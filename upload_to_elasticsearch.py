import json
import os
import re

from elasticsearch import Elasticsearch, helpers

es = Elasticsearch(hosts=["172.16.80.68:9200"], timeout=50000)


def read_json_by_folder(folder_path):
    path_list = []
    for file_path in os.listdir(folder_path):
        if file_path.endswith(".json"):
            path_list.append(file_path)
    path_list.sort(key=lambda x: int(x[:-5]))

    print('file_count:', len(path_list))
    return path_list


# 如果想用curl -s -H "Content-Type: application/x-ndjson" -XPOST localhost:9200/_bulk --data-binary "@20200629105300.json"
# 的方式传输数据到es的话就用这个方法赚一下格式
# alter_json('./20200629/physical/20200629105300.json')
def alter_json(file_path):
    title = '{ "index" : { "_index" : "physical", "_type" : "physical", "_id" : "' + file_path.split('/')[-1][:-5] + '" } }'
    file_data = ''
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # pattern : 正则中的模式字符串。 repl : 替换的字符串，也可为一个函数。 string : 要被查找替换的原始字符串。
            line = re.sub(r"[\r\n\s]", "", line)
            file_data += line
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(title)
        f.write('\n')
        f.write(file_data)
        f.write('\n')


def upload_data(file_path, es_index, es_type):
    # call the function to get the string data containing docs
    file_data = ''
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # pattern : 正则中的模式字符串。 repl : 替换的字符串，也可为一个函数。 string : 要被查找替换的原始字符串。
            line = re.sub(r"[\r\n\s]", "", line)
            file_data += line

    action = {
        '_op_type': 'index',
        '_index': es_index,
        '_id': file_path.split('/')[-1][:-5],
        '_type': es_type,
        '_source': json.loads(file_data)
    }
    # print(action)
    try:
        # use the helpers library's Bulk API to index list of Elasticsearch docs
        resp = helpers.bulk(es, [action])
        print(resp)
    except Exception as err:
        print("Elasticsearch helpers.bulk() ERROR:", err)


if __name__ == '__main__':
    pre_path = './20200629/'
    extract_type = 'virtual' # physical virtual network
    folder_path = pre_path + extract_type
    file_path_list = read_json_by_folder(folder_path)
    for file_path in file_path_list:
        upload_data(folder_path+'/'+file_path, extract_type, extract_type)
        # print(folder_path+'/'+file_path)