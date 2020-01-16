import os
import shutil
import random

# must match classes in folder structure
classes = ['bath_room', 'bed_room', 'dining_room', 'entre', 'kitchen', 'living_room']

# train, test & val must be folders in out-dir
split = {
    'train': 0.6,
    'test': 0.2,
    'val': 0.2
}

target_dir = '../../images/classes'
source_dir = '../../images/all'

def run():
    for c in classes:
        files = os.listdir('{}/{}'.format(source_dir, c))
        n_train = round(len(files)*split['train'])
        n_test = round(len(files)*split['test'])
        n_val = round(len(files)*split['val'])

        random.shuffle(files)

        for f in files[0:n_train-1]:
            shutil.copyfile('{}/{}/{}'.format(source_dir, c, f), '{}/train/{}/{}'.format(target_dir, c, f))

        for f in files[n_train:n_train+n_test-1]:
            shutil.copyfile('{}/{}/{}'.format(source_dir, c, f), '{}/test/{}/{}'.format(target_dir, c, f))

        for f in files[-n_val:]:
            shutil.copyfile('{}/{}/{}'.format(source_dir, c, f), '{}/val/{}/{}'.format(target_dir, c, f))


if __name__ == '__main__':
    run()