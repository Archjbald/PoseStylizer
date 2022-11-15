import torch.utils.data
from data.base_data_loader import BaseDataLoader

from util.random_seed import seed_worker


def CreateDataset(opt):
    dataset = None

    if opt.dataset_mode == 'keypoint':
        from data.keypoint import KeyDataset
        dataset = KeyDataset()
    elif opt.dataset_mode == 'keypoint_multi':
        from data.keypoint_multi import KeyDatasetMulti
        dataset = KeyDatasetMulti()
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


class CustomDatasetDataLoaderMulti(CustomDatasetDataLoader):
    def name(self):
        return 'CustomDatasetDataLoaderMulti'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        g = torch.Generator()
        g.manual_seed(0)
        sampler = BatchSamplerMulti(self.dataset, opt.batchSize, drop_last=True)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=int(opt.nThreads))


class BatchSamplerMulti(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, opt):
        super(BatchSamplerMulti, self).__init__(sampler, batch_size, drop_last)
        self.ratios = self.sampler.ratios
        assert len(self.ratios) >= 2
        sub_sampler = (torch.utils.data.sampler.SequentialSampler if opt.serial_batches
                       else torch.utils.data.sampler.RandomSampler)

        self.sub_samplers = [sub_sampler(dataset) for dataset in self.sampler.datasets]
        self.sub_iterators = []

    def __iter__(self):
        batch = []
        self.sub_iterators = [iter(sub_s) for sub_s in self.sub_samplers]
        for idx in range(len(self.sampler)):
            batch_ratio = (idx % self.batch_size) / self.batch_size
            for set_index, ratio in enumerate(self.ratios):
                if batch_ratio < ratio:
                    break
                else:
                    batch_ratio -= ratio
            batch.append((set_index, self.sub_iterators[set_index].__next__()))

            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
