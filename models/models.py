
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'PoseStyleNet':
        assert opt.dataset_mode in ['keypoint', 'keypoint_segmentation']
        if opt.backward == 'cut':
            from .PoseCutNet import TransferCUTModel
            model = TransferCUTModel()
        elif opt.backward == 'cycle':
            from .PoseCycleNet import TransferCycleModel
            model = TransferCycleModel()
        else:
            from .PoseStyleNet import TransferModel
            model = TransferModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
