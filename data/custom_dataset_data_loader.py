import torch.utils.data
from data.base_data_loader import BaseDataLoader

from util.random_seed import seed_worker


def CreateDataset(opt):
    dataset = None
    
    if opt.dataset_mode == 'keypoint':
        from data.keypoint import KeyDataset
        dataset = KeyDataset()
    elif opt.dataset_mode == 'keypoint_segmentation':
        from data.keypoint_segmentation import KeySegDataset
        dataset = KeySegDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        g = torch.Generator()
        g.manual_seed(0)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
