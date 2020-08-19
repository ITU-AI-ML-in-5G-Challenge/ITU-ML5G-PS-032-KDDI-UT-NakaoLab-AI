import json
from datetime import datetime, timedelta

# ixnetwork-traffic-start: 1
# node-down: 25
# node-up: 25
# interface-down: 110
# interface-up: 110
# tap-loss-start: 330
# tap-loss-stop: 330
# tap-delay-start: 330
# tap-delay-stop: 330
# ixnetwork-bgp-injection-start: 90
# ixnetwork-bgp-injection-stop: 90
# ixnetwork-bgp-hijacking-start: 45
# ixnetwork-bgp-hijacking-stop: 45
# ixnetwork-traffic-stop: 1


type_list = ['ixnetwork-traffic-start', 'node-down', 'node-up', 'interface-down', 'interface-up', 'tap-loss-start',
             'tap-loss-stop', 'tap-delay-start', 'tap-delay-stop', 'ixnetwork-bgp-injection-start',
             'ixnetwork-bgp-injection-stop', 'ixnetwork-bgp-hijacking-start', 'ixnetwork-bgp-hijacking-stop',
             'ixnetwork-traffic-stop']


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

    def get_type_code(self, file_name):
        type = self.get_type(file_name)
        return type_list.index(type)

def load_label(path):
    with open(path, 'r') as label:
        label_dict = json.load(label)
        return Recipes(label_dict["recipes"])


if __name__ == '__main__':
    recipes = load_label("./label-for-learning.json")
    print(recipes.get_type("20200629105300.json"))
    print(recipes.get_type("20200629105300.json"))
    print(recipes.get_type("20200629105400.json"))
    print(recipes.get_type("20200629105500.json"))
    print(recipes.get_type("20200629105600.json"))
    print(recipes.get_type("20200629105700.json"))
    print(recipes.get_type("20200629105800.json"))
    print(recipes.get_type("20200629105900.json"))
    print(recipes.get_type("20200629110000.json"))


    print(recipes.get_type_code("20200629105300.json"))
    print(recipes.get_type_code("20200629105300.json"))
    print(recipes.get_type_code("20200629105400.json"))
    print(recipes.get_type_code("20200629105500.json"))
    print(recipes.get_type_code("20200629105600.json"))
    print(recipes.get_type_code("20200629105700.json"))
    print(recipes.get_type_code("20200629105800.json"))
    print(recipes.get_type_code("20200629105900.json"))
    print(recipes.get_type_code("20200629110000.json"))