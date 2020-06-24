# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# main.py


from argparse import ArgumentParser
import json
import os

from make_hdf5 import make_hdf5
from train import train_framework



def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config_path', type=str, default='./configs/Table1/contra_biggan32_cifar_hinge_no.json')
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--log_output_path', type=str, default=None)

    parser.add_argument('--seed', type=int, default=0, help='seed for random number')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    
    parser.add_argument('--train_rate', type=float, default=1.0, help='control the numver of train dataset')
    parser.add_argument('--reduce_class', type=float, default=1.0, help='control the numver of classes')
    parser.add_argument('-l', '--load_all_data_in_memory', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-g', '--calculate_z_grad', action='store_true')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config['data_processing'][key] = default_value
                config['train'][key] = default_value

        for key, value in config['data_processing'].items():
            if key not in config['train']:
                config['train'][key] = value
    else:
        raise NotImplementedError
    
    hdf5_path_train = make_hdf5(**config['data_processing'], mode='train') if args.load_all_data_in_memory else None
    eval_mode = 'test' if config['data_processing']['dataset_name'] == 'cifar10' else 'valid'
    hdf5_path_valid = make_hdf5(**config['data_processing'], mode=eval_mode) if args.load_all_data_in_memory else None
    if args.train is True:
        train_framework(**config['train'], config=config['train'], hdf5_path_train=hdf5_path_train, hdf5_path_valid=hdf5_path_valid)

if __name__ == '__main__':
    main()
