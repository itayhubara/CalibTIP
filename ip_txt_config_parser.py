import argparse

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--config-file', type=str, metavar='FILE', help='path to config file')
parser.add_argument('--compression_rate', default=0, type=float, help='Index of precision configuration to get')
args = parser.parse_args()

f = open(args.config_file,"r")
for i in range(13):
    aa = f.readline()
    bb = aa.replace('\t', ', ').replace('\n','').replace('[','').replace(']','').replace("'",'').split(', ')
    if bb[0] != str(args.compression_rate):
        continue
    config_dict = {}
    for key in bb[1:]:
        config_dict[key] = [4,4]
print(config_dict)

