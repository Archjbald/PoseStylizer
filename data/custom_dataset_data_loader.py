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
        nb_img_ratio = round(round(opt.batchSize * opt.ratio_multi) / (len(self.dataset.ratios) - 1) - 0.5)
        self.nb_imgs = [nb_img_ratio for _ in self.dataset.ratios[1:]]
        self.nb_imgs.insert(0, opt.batchSize - sum(self.nb_imgs))
        sampler = BatchSamplerMulti(self.dataset, opt.batchSize, self.nb_imgs, drop_last=True, opt=opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=int(opt.nThreads))

    def __len__(self):
        limitant_dataset = 0
        limitant_value = self.dataset.total_size / 1E-6

        for d, nb in self.nb_imgs:
            if not nb:
                continue
            value = self.dataset.datasets[d].size / nb
            if value < limitant_value:
                limitant_value = value
                limitant_dataset = d

        return min(len(self.dataset.datasets[limitant_dataset]), self.opt.max_dataset_size)


class BatchSamplerMulti(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, batch_size, nb_imgs, drop_last, opt):
        super(BatchSamplerMulti, self).__init__(sampler, batch_size, drop_last)
        self.nb_imgs = nb_imgs
        assert len(self.nb_imgs) >= 2
        sub_sampler = (torch.utils.data.sampler.SequentialSampler if opt.serial_batches
                       else torch.utils.data.sampler.RandomSampler)

        self.sub_samplers = [sub_sampler(dataset) for dataset in self.sampler.datasets]
        self.sub_iterators = []

    def __iter__(self):
        batch = []
        self.sub_iterators = [iter(sub_s) for sub_s in self.sub_samplers]
        set_index = 0
        for idx in range(len(self.sampler)):
            while len(batch) >= sum(self.nb_imgs[:set_index + 1]):
                set_index += 1

            batch.append((set_index, self.sub_iterators[set_index].__next__()))

            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
