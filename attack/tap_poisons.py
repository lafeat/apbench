import torch
import datetime
import time
import attack.data.tap_base as tap_base

torch.backends.cudnn.benchmark = tap_base.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(tap_base.consts.SHARING_STRATEGY)

def tap_gen():
    parse = tap_base.options().parse_args()
    setup = tap_base.utils.system_startup(parse)
    model = tap_base.Client(parse, setup=setup)
    materials = tap_base.Furnace(
        parse, model.defs.batch_size, model.defs.augmentations, setup=setup
    )
    forgemaster = tap_base.Forgemaster(parse, setup=setup)
    start_time = time.time()

    if parse.pretrained:
        stats_clean = None
    else:
        stats_clean = model.train(materials, max_epoch=parse.max_epoch)
    train_time = time.time()

    poison_delta = forgemaster.forge(model, materials)
    forge_time = time.time()

    if parse.save is not None:
        materials.export_poison(poison_delta, path=parse.poison_path, mode=parse.save)

    print("-------------Job finished-------------")
