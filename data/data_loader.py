def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())

    if opt.shuffle and not "shuffle" in opt.pairLst:
        opt.pairLst = opt.pairLst.replace('.csv', '-shuffle.csv')
    if "shuffle" in opt.pairLst:
        opt.shuffle = True

    data_loader.initialize(opt)
    return data_loader
