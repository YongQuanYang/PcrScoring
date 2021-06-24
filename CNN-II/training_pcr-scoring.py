from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
import pickle
import random
import cv2
from model import *
from preprocess import *


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

BS = 128
NE = 256

# prepare the data for experiments
train_root = r'./train'
x_train, y_train = load_data(train_root)
print(x_train.shape)
print(y_train.shape)
num_train = x_train.shape[0]

val_root = r'./val'
x_val, y_val = load_data(val_root)
print(x_val.shape)
print(y_val.shape)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BS)

# create the base pre-trained model
model_path = r'./scl_ckpts/0000/release_representation.h5'
model_path = str(model_path)
encoder = load_model(model_path, custom_objects={'UnitNormLayer': UnitNormLayer})
rep = encoder.output
predictions = Dense(2, activation='softmax')(rep)
classification = Model(inputs=encoder.input, outputs=predictions)
cce_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

for i, layer in enumerate(classification.layers):
   print(i, layer.name)
for layer in classification.layers:
   layer.trainable = True
for layer in encoder.layers:
    layer.trainable = False


#build optimizeer
optimizer = SGD(lr=0.001, momentum=0.9)


#build train/test loss recorder
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_ACC')
val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_ACC')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_preds = classification(x, training=True)
        loss = cce_loss_obj(y, y_preds)

    gradients = tape.gradient(loss,
                              classification.trainable_variables)
    optimizer.apply_gradients(zip(gradients,
                                   classification.trainable_variables))
    train_loss(loss)
    train_acc(y, y_preds)
@tf.function
def val_step(x, y):
    y_preds = classification(x, training=False)
    v_loss = cce_loss_obj(y, y_preds)
    val_loss(v_loss)
    val_acc(y, y_preds)


trainll = []
valll = []
train_accll = []
val_accll = []
ckpt_path = './pcr-score_ckpts'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
statistic_path = './pcr-score_statistics'
if not os.path.exists(statistic_path):
    os.makedirs(statistic_path)

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range = 10,
                             zoom_range = [0.8, 1.2],
                             width_shift_range=0.2,
                             height_shift_range=0.2)
data_iter = datagen.flow(x_train, y_train, batch_size=BS)

for epoch in range(NE):
    # Reset the metrics at the start of the next epoch
    print('Training Epoch {}.....'.format(epoch))
    train_loss.reset_states()
    val_loss.reset_states()
    train_acc.reset_states()
    val_acc.reset_states()

    i = 0
    for x, y in data_iter:
        train_step(x, y)
        i = i+1
        if i > (num_train/BS):
            break

    for x_v, y_v in val_ds:
        val_step(x_v, y_v)

    template = 'Epoch {}, Loss: {}, Val Loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          val_loss.result()))
    template = 'Epoch {}, Acc: {}, Val Acc: {}'
    print(template.format(epoch + 1,
                          train_acc.result(),
                          val_acc.result()))
    trainll.append(train_loss.result())
    valll.append(val_loss.result())
    train_accll.append(train_acc.result())
    val_accll.append(val_acc.result())
    
    ckpt_epoch_dir = "%s/%04d"%(ckpt_path,epoch)
    if not os.path.isdir(ckpt_epoch_dir):
        os.makedirs(ckpt_epoch_dir)
    save_path_r = os.path.join(ckpt_epoch_dir, 'release_classification.h5')
    classification.save(save_path_r)
    
    import pickle
    f=open(os.path.join(statistic_path, 'train_test_loss.pckl'), 'wb')
    pickle.dump([trainll, valll, train_accll, val_accll], f)
    f.close()
