import argparse
import pandas as pd
import csv
import os

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--config-file', type=str, metavar='FILE', help='path to config file')
parser.add_argument('--cfg-idx', default=0, type=int, help='Index of precision configuration to get')
parser.add_argument('--column', type=str, help='path to config file') # [Configuration, state_dict_path]
args = parser.parse_args()


df = pd.read_csv(args.config_file, sep='\t')
res = df[args.column][args.cfg_idx]
print(res)
