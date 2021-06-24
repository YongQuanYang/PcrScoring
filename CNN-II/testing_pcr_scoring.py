# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
from glob import glob
import sys
from sklearn import metrics
from shutil import copyfile
from shutil import move
import pickle
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '7'  # 只使用指定GPU

sys.path.append("..")


def img_preprocess(img_path):
    img = Image.open(img_path)
    if np.asarray(img).shape[2] == 4:
        img = Image.open(img_path).convert("RGB")
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, mode='tf')
    return img


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽tf info信息

    model_path = r'./pcr-score_ckpts/0000/release_classification.h5'
    model_path = str(model_path)
    model = load_model(model_path, custom_objects={'UnitNormLayer': UnitNormLayer})
    
    source_path = './test'
    inffer_path = './test_inffer'
    if not os.path.exists(inffer_path):
        os.makedirs(inffer_path)
    
    test_dirs = os.listdir(source_path)
    for td in test_dirs:
        sdir = os.path.join(source_path, td)
        inffer_dir = os.path.join(inffer_path, td)
        if not os.path.exists(inffer_dir):
            os.makedirs(inffer_dir)
        simgs = os.listdir(sdir)
        for simg in simgs:
            img_p = os.path.join(sdir, simg)
            img = img_preprocess(img_p)
            prob_list = model.predict(img)[0]
            prob_list = prob_list.tolist()
            sps = simg.split('.')
            dimgn = sps[0]+'_'+str(prob_list[0])+'_'+str(prob_list[1])+'.png'
            destp = os.path.join(inffer_dir, dimgn)
            copyfile(img_p, destp)
