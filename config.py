# config.py
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="PACS",
        choices=["PACS", "VLCS", "OfficeHome", "TerraIncognita", "DomainNet"],
    )
    parser.add_argument("--source-domain", nargs="+", default=["photo", "cartoon", "art_painting"])
    parser.add_argument("--target-domain", nargs="+", default=["sketch"])
    parser.add_argument(
        "--known-classes",
        nargs="+",
        default=["dog", "elephant", "giraffe", "horse", "guitar", "house", "person"],
    )
    parser.add_argument("--unknown-classes", nargs="+", default=[])

    parser.add_argument("--random-split", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--algorithm", default="medic")
    parser.add_argument("--task-d", type=int, default=3)
    parser.add_argument("--task-c", type=int, default=3)
    parser.add_argument("--task-per-step", nargs="+", type=int, default=[3, 3, 3])
    parser.add_argument(
        "--weight-per-step", nargs="+", type=float, default=[1 / 2, 1 / 3, 1 / 6], help="arith only"
    )
    parser.add_argument("--selection-mode", default="random")  # random, hard

    parser.add_argument("--net-name", default="resnet50")
    parser.add_argument("--optimize-method", default="Adam")
    parser.add_argument("--schedule-method", default="StepLR")
    parser.add_argument("--num-epoch", type=int, default=5100)
    parser.add_argument("--eval-step", type=int, default=300)
    parser.add_argument(
        "--lr", type=float, default=6e-6
    )  # Alpha (meta-lr) has been calculated in the following code, so it is set to 1/t of the default learning rate.
    parser.add_argument("--meta-lr", type=float, default=1e-2)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--without-cls", action="store_true")
    parser.add_argument("--without-bcls", action="store_true")
    parser.add_argument("--share-param", action="store_true")

    parser.add_argument("--save-dir", default="./log/wxr/MEDIC-plus/save")
    parser.add_argument("--save-name", default="reproduce")
    parser.add_argument("--save-best-test", action="store_true")
    parser.add_argument("--save-later", action="store_true")

    parser.add_argument("--num-epoch-before", type=int, default=0)

    return parser.parse_args()


args = get_args()

# It can be used to replace the following code, but the editor may take it as an error.
# locals().update(vars(args))

# It can be replaced by the preceding code.
dataset = args.dataset
source_domain = sorted(args.source_domain)
target_domain = sorted(args.target_domain)
# known_classes = sorted(args.known_classes)
if args.dataset == "PACS":
    known_classes = args.known_classes = [
        "dog",
        "elephant",
        "giraffe",
        "horse",
        "guitar",
        "house",
        "person",
    ]
elif args.dataset == "OfficeHome":
    known_classes = args.known_classes = list(
        os.listdir("/workspaces/MEDIC-plus/data_list/OfficeHome_list/train/Art")
    )
elif args.dataset == "VLCS":
    known_classes = args.known_classes = ["bird", "car", "chair", "dog", "person"]
elif args.dataset == "DomainNet":
    known_classes = args.known_classes = list(
        os.listdir("/workspaces/MEDIC-plus/data_list/DomainNet_list/train/clipart")
    )
elif args.dataset == "TerraIncognita":
    known_classes = args.known_classes = list(
        os.listdir("/workspaces/MEDIC-plus/dataset_raw/TerraIncognita/location_38")
    )

unknown_classes = sorted(args.unknown_classes)
random_split = args.random_split
gpu = args.gpu
batch_size = args.batch_size = 18 if args.dataset == "DomainNet" else args.batch_size
algorithm = args.algorithm
task_d = args.task_d
task_c = args.task_c
task_per_step = args.task_per_step
weight_per_step = (
    [1 / 3, 4 / 15, 1 / 5, 2 / 15, 1 / 15] if args.dataset == "DomainNet" else args.weight_per_step
)
selection_mode = args.selection_mode
net_name = args.net_name
optimize_method = args.optimize_method
schedule_method = args.schedule_method
num_epoch = args.num_epoch = 15000 if args.dataset == "DomainNet" else args.num_epoch
eval_step = args.eval_step
lr = args.lr
meta_lr = args.meta_lr
nesterov = args.nesterov
without_cls = args.without_cls
without_bcls = args.without_bcls = True
share_param = args.share_param
save_dir = args.save_dir = os.path.join(
    args.save_dir, args.algorithm, args.dataset, args.target_domain[0]
)
save_name = args.save_name
save_later = args.save_later
save_best_test = args.save_best_test
num_epoch_before = args.num_epoch_before
crossval = True

# DomainBed
if dataset == "PACS":
    train_dir = "/workspaces/MEDIC-plus/data_list/PACS_list/train"
    val_dir = "/workspaces/MEDIC-plus/data_list/PACS_list/crossval"
    test_dir = [
        "/workspaces/MEDIC-plus/data_list/PACS_list/train",
        "/workspaces/MEDIC-plus/data_list/PACS_list/crossval",
    ]
    sub_batch_size = batch_size // 2
    small_img = False
elif dataset == "OfficeHome":
    train_dir = "/workspaces/MEDIC-plus/data_list/OfficeHome_list/train"
    val_dir = "/workspaces/MEDIC-plus/data_list/OfficeHome_list/crossval"
    test_dir = [
        "/workspaces/MEDIC-plus/data_list/OfficeHome_list/train",
        "/workspaces/MEDIC-plus/data_list/OfficeHome_list/crossval",
    ]
    sub_batch_size = batch_size // 2
    small_img = False
elif dataset == "VLCS":
    train_dir = "/workspaces/MEDIC-plus/data_list/VLCS_list/train"
    val_dir = "/workspaces/MEDIC-plus/data_list/VLCS_list/crossval"
    test_dir = [
        "/workspaces/MEDIC-plus/data_list/VLCS_list/train",
        "/workspaces/MEDIC-plus/data_list/VLCS_list/crossval",
    ]
    sub_batch_size = batch_size
    small_img = False
elif dataset == "TerraIncognita":
    train_dir = "/workspaces/MEDIC-plus/data_list/TerraIncognita_list/train"
    val_dir = "/workspaces/MEDIC-plus/data_list/TerraIncognita_list/crossval"
    test_dir = [
        "/workspaces/MEDIC-plus/data_list/TerraIncognita_list/train",
        "/workspaces/MEDIC-plus/data_list/TerraIncognita_list/crossval",
    ]
    sub_batch_size = batch_size
    small_img = False
elif dataset == "DomainNet":
    train_dir = "/workspaces/MEDIC-plus/data_list/DomainNet_list/train"
    val_dir = "/workspaces/MEDIC-plus/data_list/DomainNet_list/crossval"
    test_dir = [
        "/workspaces/MEDIC-plus/data_list/DomainNet_list/train",
        "/workspaces/MEDIC-plus/data_list/DomainNet_list/crossval",
    ]
    sub_batch_size = batch_size // 2
    small_img = False

for d in ["", "log", "param", "model/val", "model/test"]:
    os.makedirs(os.path.join(save_dir, d), exist_ok=True)

log_path = os.path.join(save_dir, "log", save_name + "_train.txt")
param_path = os.path.join(save_dir, "param", save_name + ".pkl")
model_val_path = os.path.join(save_dir, "model", "val", save_name + ".tar")
model_test_path = os.path.join(save_dir, "model", "test", save_name + ".tar")
renovate_step = int(num_epoch * 0.85) if save_later else 0

assert task_d * task_c == sum(task_per_step)
