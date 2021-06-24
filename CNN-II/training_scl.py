from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import load_data
import tensorflow as tf
from model import *
import numpy as np


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#supervised contrastive learning loss
def supervised_nt_xent_loss(z, y, temperature=0.5, base_temperature=0.07):
    batch_size = tf.shape(z)[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = tf.expand_dims(y, -1)

    # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
    #     has the same class as sample i. Can be asymmetric.
    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )
    # # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    # # tile mask
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(
        mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)
    return loss


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

BS = 256
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

#build encoder
input_tensor = Input(shape=(128, 128, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
# add a global spatial average pooling layer and normalize it
x = base_model.output
x = GlobalAveragePooling2D()(x)
representation = UnitNormLayer()(x)
encoder = Model(inputs=base_model.input, outputs=representation)

for i, layer in enumerate(encoder.layers):
   print(i, layer.name)

for layer in encoder.layers:
   layer.trainable = True


#build optimizeer
optimizer = SGD(lr=0.001, momentum=0.9)


#build train/val loss recorder
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')


#train step for scl based representation learning
@tf.function
def train_step_r(x, y):
    with tf.GradientTape() as tape:
        r = encoder(x, training=True)
        loss = supervised_nt_xent_loss(r, y)

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)


#test step for scl based representation learning
@tf.function
def val_step_r(x, y):
    r = encoder(x, training=False)
    v_loss = supervised_nt_xent_loss(r, y)
    val_loss(v_loss)

trainll = []
valll = []
ckpt_path = './scl_ckpts'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
statistic_path = './scl_statistics'
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

    i = 0
    for x, y in data_iter:
        train_step_r(x, y)
        i = i+1
        if i > (num_train/BS):
            break

    for x_v, y_v in val_ds:
        val_step_r(x_v, y_v)

    template = 'Epoch {}, Loss: {}, Val Loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          val_loss.result()))
    trainll.append(train_loss.result())
    valll.append(val_loss.result())
    
    ckpt_epoch_dir = "%s/%04d"%(ckpt_path,epoch)
    if not os.path.isdir(ckpt_epoch_dir):
        os.makedirs(ckpt_epoch_dir)
    save_path_r = os.path.join(ckpt_epoch_dir, 'release_representation.h5')
    encoder.save(save_path_r)
    
    import pickle
    f=open(os.path.join(statistic_path, 'train_test_loss.pckl'), 'wb')
    pickle.dump([trainll, valll], f)
    f.close()

