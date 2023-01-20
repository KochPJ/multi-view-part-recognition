import os
import random
from shutil import copyfile
import random

if __name__ == '__main__':
    src = '/mnt/HDD/industrial_part_recognition/'
    dst = '/mnt/HDD/to_seg/industrial_part_recognition'

    classes = {}
    sets_per_cls = 4
    for root, dirs, files in os.walk(src):
        for file in files:
            if '_rgb.png' in file and 'test_data' in root:
                new_root = os.path.join(dst, root[len(src):])
                cls = root[len(src):].split('/')[0]
                if cls not in classes:
                    classes[cls] = []


                classes[cls].append((root, new_root, file))

    print('num images {}'.format(sum([len(v) for v in classes.values()])))
    new = 1
    for cls in classes.keys():
        sets = classes[cls]
        random.shuffle(sets)
        classes[cls] = sets[:sets_per_cls]
        for root, new_root, file in sets[:sets_per_cls]:
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            print(new, os.path.join(new_root, file))
            copyfile(os.path.join(root, file), os.path.join(new_root, file))
            new += 1

    print('num images {}'.format(sum([len(v) for v in classes.values()])))



