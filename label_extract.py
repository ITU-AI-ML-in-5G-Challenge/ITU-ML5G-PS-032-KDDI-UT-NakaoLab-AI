import json
from datetime import datetime, timedelta


class Recipes:
    def __init__(self, recipes):
        self.recipes = recipes

    def get_type(self, file_name):
        date_time = datetime.strptime(file_name, '%Y%m%d%H%M00.json')
        for json_obj in self.recipes:
            started_at = datetime.strptime(json_obj["started_at"], '%Y-%m-%dT%H:%M:%S') + timedelta(hours=9)
            stopped_at = datetime.strptime(json_obj["stopped_at"], '%Y-%m-%dT%H:%M:%S') + timedelta(hours=9)
            next_started_at = stopped_at + timedelta(minutes=5)
            if started_at < date_time < next_started_at:
                return json_obj["type"]


def load_label(path):
    with open(path, 'r') as label:
        label_dict = json.load(label)
        return Recipes(label_dict["recipes"])


if __name__ == '__main__':
    recipes = load_label("./label-for-learning.json")
    print(recipes.get_type("20200629105300.json"))
    print(recipes.get_type("20200629105400.json"))
    print(recipes.get_type("20200629105500.json"))
    print(recipes.get_type("20200629105600.json"))
    print(recipes.get_type("20200629105700.json"))
    print(recipes.get_type("20200629105800.json"))
    print(recipes.get_type("20200629105900.json"))
    print(recipes.get_type("20200629110000.json"))
