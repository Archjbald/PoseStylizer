def CreateDataLoader(opt):
    if '_multi' in opt.dataset_mode:
        if opt.phase == 'train':
            from data.custom_dataset_data_loader import CustomDatasetDataLoaderMulti
            data_loader = CustomDatasetDataLoaderMulti()
        else:
            from data.custom_dataset_data_loader import CustomDatasetDataLoader
            data_loader = CustomDatasetDataLoader()
    else:
        from data.custom_dataset_data_loader import CustomDatasetDataLoader
        data_loader = CustomDatasetDataLoader()
    print(data_loader.name())

    data_loader.initialize(opt)
    return data_loader
