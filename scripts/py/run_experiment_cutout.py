import argparse
import json
import os
import subprocess


def set_experiment_default_args(parser):
    parser.add_argument('--expname', '-e', default="tmp", type=str, help='experiment name')
    parser.add_argument('--strategy', '-s', default="nofilter", type=str, help='nofilter, sb, kath')
    parser.add_argument('--calculator', '-c', default="relative", type=str, help='relative, random')
    parser.add_argument('--fp_selector', '-f', default="alwayson", type=str, help='alwayson, stale')
    parser.add_argument('--dataset', '-d', default="cifar10", type=str, choices=['svhn', 'cifar10', 'cifar100'])
    parser.add_argument('--custom-lr', default=None, type=str)
    parser.add_argument('--accelerate-lr', dest='accelerate_lr', action='store_true',
                        help='Use hardcoded accelerated lr schedule')
    parser.add_argument('--profile', dest='profile', action='store_true',
                        help='turn profiling on')
    parser.add_argument('--num-trials', default=1, type=int, help='number of trials')
    parser.add_argument('--batch-size', '-b', default=128, type=int, help='batch size')
    parser.add_argument('--src-dir', default="./", type=str, help='/path/to/pytorch-cifar')
    #parser.add_argument('--dst-dir', default="/ssd/ahjiang/output/", type=str, help='/path/to/dst/dir')
    parser.add_argument('--dst-dir', default="/proj/BigLearning/ahjiang/output/", type=str, help='/path/to/dst/dir')
    return parser

class Seeder():
    def __init__(self):
        self.seed = 1336

    def get_seed(self):
        self.seed += 1
        return self.seed

def get_sampling_min():
    return 0

def get_decay():
    return 0.0005

def get_max_history_length():
    return 1024

def get_kath_strategy():
    return "biased"

def get_num_epochs(dataset, profile):
    if profile:
        return 3
    if dataset == "svhn":
        return 160
    else:
        return 200

def get_learning_rate(dataset, accelerate_lr, custom_lr):
    if custom_lr is not None:
        return custom_lr

    base = "/home/ahjiang/Cutout/pytorch-cifar/data/config/sysml20/"
    base = "/users/ahjiang/src/Cutout/pytorch-cifar/data/config/sysml20/"
    if accelerate_lr:
        if dataset == "svhn":
            return "{}/svhn/sampling-relative_svhn_wideresnet_0_128_1024_0.0005_trial1_seed1337_v4.lr".format(base)
        elif dataset == "cifar10":
            return "{}/cifar10/sampling_cifar10_wideresnet_0_128_1024_0.0005_trial1_seed1337_v4.lr".format(base)
        elif dataset == "cifar100":
            return "{}/cifar100/sampling-relative_cifar100_wideresnet_0_128_1024_0.0005_trial1_seed1337_v4.lr".format(base)
    else:
        if dataset == "svhn":
            return "{}/svhn/lr_sched_svhn_wideresnet".format(base)
        elif dataset == "cifar10":
            return "{}/cifar10/lr_sched_cifar10_wideresnet".format(base)
        elif dataset == "cifar100":
            return "{}/cifar100/lr_sched_cifar100_wideresnet".format(base)

def get_length(dataset):
    if dataset == "cifar10":
        return 16
    elif dataset == "cifar100":
        return 8
    elif dataset == "svhn":
        return 20

def get_sample_size(batch_size):
    return batch_size * 3

def get_model():
    return "wideresnet"

def get_output_dirs(dst_dir):
    pickles_dir = os.path.join(dst_dir, "pickles")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if not os.path.exists(pickles_dir):
        os.mkdir(pickles_dir)
    return dst_dir, pickles_dir

def get_output_files(strategy,
                     calculator,
                     fp_selector,
                     dataset,
                     net,
                     sampling_min,
                     batch_size,
                     max_history_length,
                     decay,
                     trial,
                     seed,
                     kath_strategy,
                     static_sample_size):

    if strategy == "kath":
        identifier = "kath-{}".format(kath_strategy)
        max_history_length = static_sample_size
    elif strategy == "nofilter":
        identifier = "nofilter"
    elif strategy == "topk":
        identifier = "topk"
        max_history_length = static_sample_size
    elif strategy == "sb":
        identifier = "{}-{}-{}".format(strategy, calculator, fp_selector)

    output_file = "{}_{}_{}_{}_{}_{}_{}_trial{}_seed{}_v5".format(identifier,
                                                                  dataset,
                                                                  net,
                                                                  sampling_min,
                                                                  batch_size,
                                                                  max_history_length,
                                                                  decay,
                                                                  trial,
                                                                  seed)

    pickle_file = "{}_{}_{}_{}_{}_{}_{}_trial{}_seed{}".format(identifier,
                                                               dataset,
                                                               net,
                                                               sampling_min,
                                                               batch_size,
                                                               max_history_length,
                                                               decay,
                                                               trial,
                                                               seed)
    return output_file, pickle_file

def get_experiment_dirs(dst_dir, dataset, expname):
    output_dir = os.path.join(dst_dir, dataset, expname)
    pickles_dir = os.path.join(output_dir, "pickles")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(pickles_dir):
        os.mkdir(pickles_dir)
    return output_dir, pickles_dir

def main(args):
    seeder = Seeder()
    src_dir = os.path.abspath(args.src_dir)
    sampling_min = get_sampling_min()
    decay = get_decay()
    static_sample_size = get_sample_size(args.batch_size)
    lr_file = get_learning_rate(args.dataset, args.accelerate_lr, args.custom_lr)
    length = get_length(args.dataset)
    model = get_model()
    num_epochs = get_num_epochs(args.dataset, args.profile)
    kath_strategy = get_kath_strategy()
    max_history_length = get_max_history_length()
    output_dir, pickles_dir = get_experiment_dirs(args.dst_dir, args.dataset, args.expname)
    assert(args.strategy in ["nofilter", "sb", "kath"])

    for trial in range(1, args.num_trials+1):
        seed = seeder.get_seed()
        output_file, pickle_file = get_output_files(args.strategy,
                                                    args.calculator,
                                                    args.fp_selector,
                                                    args.dataset,
                                                    model,
                                                    sampling_min,
                                                    args.batch_size,
                                                    max_history_length,
                                                    decay,
                                                    trial,
                                                    seed,
                                                    kath_strategy,
                                                    static_sample_size)
        if args.profile:
            cmd = "python -m cProfile -o {}/{}.prof train.py ".format(output_dir, args.expname)
        else:
            cmd = "python train.py "
        if args.dataset in ["cifar10", "cifar100"]:
            cmd += "--data_augmentation "
        cmd += "--dataset {} ".format(args.dataset)
        cmd += "--model {} ".format(model)
        cmd += "--lr_sched {} ".format(lr_file)
        cmd += "--length {} ".format(length)
        cmd += "--epochs {} ".format(num_epochs)
        cmd += "--batch_size {} ".format(args.batch_size)
        cmd += "--cutout "
        cmd += "--forwardlr "
        cmd += "--sb "
        cmd += "--strategy {} ".format(args.strategy)
        cmd += "--calculator {} ".format(args.calculator)
        cmd += "--fp_selector {} ".format(args.fp_selector)

        cmd = cmd.strip()

        output_path = os.path.join(output_dir, output_file)
        print("========================================================================")
        print(cmd)
        print("------------------------------------------------------------------------")
        print(output_path)

        with open(os.path.join(pickles_dir, output_file) + "_cmd", "w+") as f:
            f.write(cmd)

        cmd_list = cmd.split(" ")
        with open(output_path, "w+") as f:
            subprocess.call(cmd_list, stdout=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser = set_experiment_default_args(parser)
    args = parser.parse_args()
    main(args)
