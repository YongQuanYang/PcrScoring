# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import pickle
from PIL import Image
import random
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical


def load_data(data_root):
    pkl_path = os.path.join(data_root, 'data_cls.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            [data, label] = pickle.load(f)
    else:
        dir_names = ['nonpcr', 'pcr']
        data_label = []
        for i in range(len(dir_names)):
            img_dir = os.path.join(data_root, dir_names[i])
            ds_path = os.path.join(img_dir, '.DS_Store')
            if os.path.exists(ds_path):
                os.remove(ds_path)
            img_names = os.listdir(img_dir)

            for img_name in img_names:
                img_path = os.path.join(img_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((128, 128))
                if np.asarray(img).shape[2] == 4:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((128, 128))
                if np.asarray(img).shape != (128, 128, 3):
                    print('shape_error: {}'.format(img_path))
                    continue
                try:
                    print(img_path)
                except:
                    print('Name error!')
                    continue
                img = np.array(img)
                data_label.append((img, i))

        random.shuffle(data_label)
        data, label = [], []
        for v in data_label:
            data.append(v[0])
            label.append(v[1])
        with open(pkl_path, 'wb') as f:
            pickle.dump([data, label], f)

    data = preprocess_input(np.array(data), mode='tf')
    label = np.array(label)
    return data, label


if __name__ == '__main__':
    train_root = r'./test'
    load_data(train_root)
