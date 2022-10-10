
def create_model(opt):
    model = None
    print(opt.model)

    assert opt.dataset_mode in ['keypoint', 'keypoint_segmentation', 'keypoint_multi']
    if opt.model == 'PoseStyleNet':
        if opt.backward == 'cut':
            from .PoseCutNet import TransferCUTModel
            model = TransferCUTModel()
        elif opt.backward == 'cycle':
            from .PoseCycleNet import TransferCycleModel
            model = TransferCycleModel()
        elif opt.backward == 'cycle_hpe':
            from .PoseCycleHPENet import TransferCycleHPEModel
            model = TransferCycleHPEModel()
        elif opt.backward == 'cycle_wgan':
            from .PoseCycleWGANet import TransferCycleWGANModel
            model = TransferCycleWGANModel()
        elif opt.backward == 'better_cycle':
            from .PoseBetterCycleNet import TransferBetterCycleModel
            model = TransferBetterCycleModel()
        else:
            from .PoseStyleNet import TransferModel
            model = TransferModel()
    elif opt.model == 'PATNCycle':
        from .PATNCycle import PATNCycle
        model = PATNCycle()
    elif opt.model == 'UCCPT':
        from .UCCPT import UCCPT
        model = UCCPT()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    print("Model [%s] was created" % (model.name()))
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}")
    submodels = {sub: model.__getattr__(sub) for sub in model.model_names}
    print("\n".join([f"\t\t{name}: {sum(p.numel() for p in sub.parameters())}" for name, sub in submodels.items()]))
    return model
