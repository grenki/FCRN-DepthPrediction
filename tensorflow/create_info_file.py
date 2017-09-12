import os
import shutil

source_folder = '/dtd/datasets/spectando_heatmaps_2/'
test_path = source_folder + 'test.txt'
train_path = source_folder + 'train.txt'
res_folder = '/dtd/datasets/spectando_with_depth/'

if not os.path.exists(res_folder):
    os.mkdir(res_folder)

shutil.copy(test_path, res_folder + 'test.txt')
shutil.copy(train_path, res_folder + 'train.txt')

data = open(test_path).read() + open(train_path).read()
info = (s.split(' ') for s in data.split('\n'))
for s in info:
    s[0] = source_folder + s[0]
    s[1] = res_folder + s[0]

info_string = '\n'.join(' '.join(s) for s in info)
open('info.txt', 'w').write(info_string)
