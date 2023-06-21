import torch
import datetime
import time
import village

torch.backends.cudnn.benchmark = village.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(village.consts.SHARING_STRATEGY)
args = village.options().parse_args()

if __name__ == "__main__":
    setup = village.utils.system_startup(args)
    model = village.Client(args, setup=setup)
    materials = village.Furnace(
        args, model.defs.batch_size, model.defs.augmentations, setup=setup
    )
    forgemaster = village.Forgemaster(args, setup=setup)
    start_time = time.time()

    if args.pretrained:
        stats_clean = None
    else:
        stats_clean = model.train(materials, max_epoch=args.max_epoch)
    train_time = time.time()

    poison_delta = forgemaster.forge(model, materials)
    forge_time = time.time()

    if args.save is not None:
        materials.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print("---------------------------------------")
    print(
        f"Finished with time: {str(datetime.timedelta(seconds=train_time - start_time))}"
    )
    print(f" forge time: {str(datetime.timedelta(seconds=forge_time - train_time))}")
    print("-------------Job finished-------------")
