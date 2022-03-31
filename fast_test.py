import sys
import os
from collections import namedtuple

from test import main as test

LOG_PATH = 'logs/'
CHECKPOINT_PATH = 'checkpoint/'


def get_options(name):
    with open(os.path.join(LOG_PATH, name, 'options.txt')) as f:
        options = f.read().split('\n')
    options = dict([line.split(': ') for line in options if line])
    for k, v in options.items():
        try:
            options[k] = int(v)
            continue
        except:
            pass
        try:
            options[k] = float(v)
            continue
        except:
            pass
        # if "," in v:
        #     try:
        #         options[k] = [int(i) for i in v.split(',')]
        #         continue
        #     except:
        #         pass
        if v == 'False':
            options[k] = False
            continue
        if v == 'True':
            options[k] = True
            continue

    return options


class ObjectOpt:
    def __init__(self, dict_options):
        for k, v in dict_options.items():
            setattr(self, k, v)


if __name__ == '__main__':
    assert len(sys.argv) >= 2, 'No name mentioned'
    options = get_options(sys.argv[1])

    object_options = ObjectOpt(options)
    object_options.max_dataset_size = 100
    object_options.random = False
    object_options.shuffle = False

    test(opt=object_options)
