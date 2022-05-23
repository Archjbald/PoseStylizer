import os
import torch
import numpy
import random


def seed_all(seed=None):
    if not seed:
        seed = random.randint(0, 2 ** 32 - 1)

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    return seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    # print("[ Using Seed : ", worker_seed, " ]")
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
