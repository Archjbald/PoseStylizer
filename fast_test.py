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
        if ',' in v:
            if v[0] == '[':
                v = v[1:-1].replace(' ', '')
            try:
                options[k] = [int(i) for i in v.split(',')]
                continue
            except:
                pass
        # if v[0] == '[':
        #     options[k] = ','.join(v[1:-1].split(', '))
        #     continue
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

    for k, v in [('ntest', float("inf")), ('results_dir', './results/'), ('aspect_ratio', 1.0), ('phase', 'test'),
                 ('which_epoch', 'latest'), ('how_many', 100)]:
        if k not in options:
            options[k] = v

    object_options = ObjectOpt(options)

    object_options.random = False
    object_options.shuffle = False

    test(opt=object_options)
