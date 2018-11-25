# encoding=utf-8

import json

class Config(object):
    def __init__(self):
        with open("perrec/config.json") as f:
            conf = json.load(f)
        self.data_dir = conf['data_dir']
        self.app_system_perm_file = conf['app_system_perm_file']
        self.app_api_file = conf['app_api_file']
        self.glove_embedding_file = conf['glove_embedding_file']
        self.api_embedding_file = conf['api_embedding_file']
        self.api_desc_token_file = conf['api_desc_token_file']
        self.sim_dump_file = conf['sim_dump_file']

CONFIG = Config()
