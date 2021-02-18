import argparse

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--config-file', type=str, metavar='FILE', help='path to config file')
parser.add_argument('--compression_rate', default=0, type=float, help='Index of precision configuration to get')
args = parser.parse_args()

config_dict = {}
f = open(args.config_file,"r")
for i in range(13):
    aa = f.readline()
    bb = aa.replace('\t', ', ').replace('    ', ', ').replace('\n','').replace('[','').replace(']','').replace("'",'').split(', ')
    try:
        cmp = float(bb[0])
        if cmp != args.compression_rate:
            continue
    except ValueError:
        continue
    for key in bb[1:]:
        config_dict[key] = [4,4]
print(config_dict)

