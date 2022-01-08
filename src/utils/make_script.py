import argparse
import os
import difflib
import collections
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="directory that includes checkpoints", required=True)
    parser.add_argument("--cfg_path", type=str, help="directory that includes config files", required=True)
    parser.add_argument("--dataset_path", type=str, help="directory that includes dataset files", required=True)
    parser.add_argument("--dataset", type=str, help="dataset in [CIFAR10, Tiny_ImageNet, ImageNet, AFHQ, CUB200", required=True)
    parser.add_argument("--num_eval", type=int, help="how many time to repeat", required=True)
    args = parser.parse_args()
    return args

def preprocess(parsed_args):
    ckpt_path = parsed_args.ckpt_path
    ckpt_list = os.listdir(ckpt_path)
    for item in ckpt_list:
        item_ = ""
        if "ACGAN" in item and "ReACGAN" not in item and "ACGAN-Mod" not in item:
            item_ = "ACGAN-Mod"+item[item.index("-"):]
        elif "BigGAN-train" in item:
            item_ = "BigGAN-Mod"+item[item.index("-"):]
        elif "ICRGAN(P)" in item:
            item_ = "BigGAN-Mod-ICR"+item[item.index("-"):]
        elif "ICRGAN(C)" in item:
            item_ = "ContraGAN-ICR"+item[item.index("-"):]
        elif "CRGAN(P)" in item:
            item_ = "BigGAN-Mod-CR"+item[item.index("-"):]
        elif "CRGAN(C)" in item:
            item_ = "ContraGAN-CR"+item[item.index("-"):]
        elif "DiffAugGAN(P)" in item:
            item_ = "BigGAN-Mod-DiffAug"+item[item.index("-"):]
        elif "DiffAugGAN(C)" in item:
            item_ = "ContraGAN-DiffAug"+item[item.index("-"):]
        elif "CCMGAN2048" in item:
            item_ = "ReACGAN-2048"+item[item.index("-"):]
        elif "CCMGAN256" in item:
            item_ = "ReACGAN-256"+item[item.index("-"):]
        elif "CCMGAN-train-2021_04_28_12_09_23" in item:
            item_ = "ReACGAN-train-2021_04_28_12_09_23"
        elif "CCMGAN-train-2021_05_03_12_19_16" in item:
            item_ = "ReACGAN-CR-train-2021_05_03_12_19_16"
        elif "CCMGAN-train-2021_05_03_12_20_37" in item:
            item_ = "ReACGAN-DiffAug-train-2021_05_03_12_20_37"
        elif "CCMGAN-train-2021_04_26_11_14_49" in item:
            item_ = "ReACGAN-train-2021_04_26_11_14_49"
        elif "CCMGAN-train-2021_05_03_19_37_26" in item:
            item_ = "ReACGAN-CR-train-2021_05_03_19_37_26"
        elif "CCMGAN-train-2021-05_03_15_36_29" in item:
            item_ = "ReACGAN-DiffAug-train-2021-05_03_15_36_29"
        if item_ != "":
            os.rename(os.path.join(ckpt_path, item), os.path.join(ckpt_path, item_))

def get_proper_configs(parsed_args):
    ckpt_path, cfg_path = parsed_args.ckpt_path, parsed_args.cfg_path
    ckpt_list, cfg_list = sorted(os.listdir(ckpt_path)), sorted(os.listdir(cfg_path))
    ckpt_cfg_dict = collections.defaultdict(str)
    for item in ckpt_list:
        ckpt_cfg_dict[os.path.join(ckpt_path,item)] = os.path.join(cfg_path, difflib.get_close_matches(item[:item.index("-train")], cfg_list, n=1, cutoff=0)[0])
    return ckpt_cfg_dict

def write_script(ckpt_cfg_dict, parsed_args):
    with open("eval_"+parsed_args.dataset+".sh", "w") as f:
        prefix_style = "CUDA_VISIBLE_DEVICES=0 python src/main.py -hdf5 -l -metrics is fid prdc" + " -data " + parsed_args.dataset_path + " -save ../studiogan/"+ parsed_args.dataset + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") + " --num_eval " + str(parsed_args.num_eval)
        if parsed_args.dataset == "ImageNet":
            prefix_ = "CUDA_VISIBLE_DEVICES=0 python src/main.py -hdf5 -l -std_stat -std_max 256 -std_step 256 -metrics is fid prdc" + " -data " + parsed_args.dataset_path + " -save ../studiogan/"+ parsed_args.dataset + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") + " --num_eval " + str(parsed_args.num_eval)
            valid = " -ref valid"
        else:
            prefix_ = "CUDA_VISIBLE_DEVICES=0 python src/main.py -hdf5 -l -batch_stat -metrics is fid prdc" + " -data " + parsed_args.dataset_path + " -save ../studiogan/"+ parsed_args.dataset + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")+ " --num_eval " + str(parsed_args.num_eval)
            valid = " -ref test"
        train, legacy, clean = " -ref train", " --resize_fn legacy", " --resize_fn clean"
        for (key, value) in ckpt_cfg_dict.items():
            others = " -ckpt " + key + " -cfg " + value + " \n"
            if "Style" in key:
                prefix = prefix_style
            else:
                prefix = prefix_
            f.write(prefix+valid+legacy+others)
            f.write(prefix+valid+clean+others)
            f.write(prefix+train+legacy+others)
            f.write(prefix+train+clean+others)

if __name__ == "__main__":
    parsed_args = parse_arguments()
    assert parsed_args.dataset in parsed_args.ckpt_path, "is it a correct dataset?"
    assert parsed_args.dataset in parsed_args.cfg_path, "is it a correct dataset"
    preprocess(parsed_args)
    ckpt_cfg_dict = get_proper_configs(parsed_args)
    write_script(ckpt_cfg_dict, parsed_args)