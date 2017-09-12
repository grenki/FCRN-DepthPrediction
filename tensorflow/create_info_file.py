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

data = open(test_path).read() + '\n' + open(train_path).read()
info = [s.split(' ') for s in data.split('\n')]


def create_folders(string):
    split = string.split('/')
    base = res_folder
    for p in split:
        if p.endswith('.png'):
            continue
        base += p + '/'
        if not os.path.exists(base):
            os.mkdir(base)


for s in info:
    if not s[0]:
        continue
    string = s[0]
    create_folders(string)
    s[0] = source_folder + string
    if len(s) == 2:
        s[1] = res_folder + string
    else:
        print s

info_string = '\n'.join(' '.join(s) for s in info)
open('info.txt', 'w').write(info_string)
