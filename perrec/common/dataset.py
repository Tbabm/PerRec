# encoding=utf-8

import random
import json
import numpy as np
from os.path import join as _

from perrec.config import CONFIG

def get_perm_num(data_dir=CONFIG.data_dir):
    app_system_perm_file = CONFIG.app_system_perm_file
    app_perm_file = _(data_dir, app_system_perm_file)
    with open(app_perm_file, 'r') as f:
        app_perms = json.load(f)
    perms = set()
    for app, perm in app_perms.items():
        perms.update(perm)
    return len(perms)

def prepare_shuffled_dataset():
    random.seed(1019)
    dataset = load_dataset()
    print("Dataset size:", len(dataset))
    random.shuffle(dataset.apps)
    print("The first app:", dataset.apps[0].name)
    return dataset

def load_dataset(data_dir=CONFIG.data_dir):
    app_api_file = CONFIG.app_api_file
    app_system_perm_file = CONFIG.app_system_perm_file
    return load_dataset_raw(data_dir, app_api_file, app_system_perm_file)

def load_dataset_raw(data_dir=CONFIG.data_dir, app_api_file="app_apis.json",
                     app_perm_file="app_system_permissions.json"):
    with open(_(data_dir, app_api_file), 'r') as f:
        app_apis = json.load(f)
    with open(_(data_dir, app_perm_file), 'r') as f:
        app_perms = json.load(f)

    app_dict = {}

    for app in app_apis.keys():
        # use at least one api, require at least one permission and have readme file
        if (app in app_perms) and app_apis[app] and app_perms[app]:
            app_dict[app] = {
                'apis': app_apis[app],
                'perms': app_perms[app],
            }
    dataset = Dataset(app_dict)
    return dataset

def load_json_file(json_file):
    with open(json_file, 'r') as f:
        json_obj = json.load(f)
    return json_obj

###############################################################################

class App(object):
    def __init__(self, name, apis, perms):
        """Initialize an app

        Args:
            apis (List(String)): A list of apis used by this app
            perms (List(String)): A list of permissions required by this app
            readme (String): Readme file of this app
        """
        self.name = name
        self.apis = apis
        self.perms = perms

###############################################################################

class Dataset(object):
    def __init__(self, apps):
        """Load and manage dataset
        
        Args:
            app_dict (List(Dict)): List of app dict, each element use app_id as key,
        values is also a dict, which contains the api list and permission list of the app
        """
        self.apps = []
        if isinstance(apps, dict):
            for name, info in apps.items():
                cur_app = App(name, info['apis'], info['perms'])
                self.apps.append(cur_app)
        elif isinstance(apps, list):
            self.apps = apps
        self.current = 0

    def __len__(self):
        return len(self.apps)

    def __getitem__(self, index):
        return self.apps[index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.apps):
            raise StopIteration
        else:
            self.current += 1
            return self.apps[self.current - 1]

    @staticmethod
    def extract_perm_set(app_perms):
        perm_set = set()
        for perms in app_perms:
            perm_set.update(perms)
        return sorted(list(perm_set))
    
    @staticmethod
    def extract_api_set(app_apis):
        api_set = set()
        for apis in app_apis:
            api_set.update(apis)
        return sorted(list(api_set))
    
    @staticmethod
    def extract_app_perm_lists(apps):
        return [app.perms for app in apps]
    
    def extract_perm_lists(self):
        return [app.perms for app in self.apps]

    @staticmethod
    def extract_app_api_lists(apps):
        return [app.apis for app in apps]
    
    def extract_api_lists(self):
        return [app.apis for app in self.apps]

def output_dataset_stat():
    dataset = load_dataset()
    api_lists = dataset.extract_api_lists()
    perm_lists = dataset.extract_perm_lists()
    api_nums = np.array([len(api_list) for api_list in api_lists])
    perm_nums = np.array([len(perm_list) for perm_list in perm_lists])
    print("Mean API num", api_nums.mean(), "+-", api_nums.std())
    print("Mean Perm num", perm_nums.mean(), "+-", perm_nums.std())

if __name__ == '__main__':
    output_dataset_stat()
