from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

import keras
import keras.backend as K
from keras import callbacks as kcb
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical

from data.dataset.market1501 import Market1501
from data.sampler import RandomSampler
from data.datagen import DataGen
from data.valdatagen import ValDataGen
from data.preprocess import imagenet_process
from backbone import ResNet50
from tripletloss import triplet_loss
from evaluator import Evaluator
from logger import setup_logger

print('version of tensorflow: {}'.format(tf.VERSION))
print('version of keras: {}'.format(keras.__version__))


''' global variables '''
g_data_root  = '/home/h_lai/Documents/dl/reid/triplet/datasets'
g_output_dir = './output'

g_num_ids  = 16
g_num_imgs = 4
g_img_h    = 256
g_img_w    = 128
g_img_ch   = 3

g_epochs   = 120
g_margin   = 0.3
g_base_lr  = 3.5e-5
g_stride   = 1

dataset   = Market1501(root=g_data_root)
t_datagen = DataGen(dataset.train, g_num_ids, g_num_imgs, g_img_w, g_img_h)
v_datagen = ValDataGen(dataset.query, dataset.gallery, (g_img_h, g_img_w))

g_num_classes = dataset.num_train_pids
g_batch_size  = g_num_ids * g_num_imgs
g_steps_per_epoch = t_datagen.sampler.len // g_batch_size


g_train_logger = setup_logger('train', g_output_dir)
g_test_logger = setup_logger('test', g_output_dir)

''' data '''
class TrainDataGenWrapper:
    def __init__(self, flow_func, dummy, num_classes):
        self.flow_func = flow_func
        self.dummy = dummy
        self.nc = num_classes

    def flow(self):
        while True:
            train_x, train_y = self.flow_func()
            train_y = to_categorical(train_y, self.nc)
            # yield train_x, [train_y, self.dummy] # for model training
            yield train_x, train_y  # for id_model training


class ValDataGenWrapper:
    def __init__(self, flow_func, dummy, num_classes):
        self.flow_func = flow_func
        self.dummy = dummy
        self.num_classes = num_classes

    def flow(self):
        while True:
            imgs, pids, camids = self.flow_func()
            yield imgs, [[pids, camids], self.dummy]


''' loss '''
factory = {
    'triplet_loss': triplet_loss(g_num_ids, g_num_imgs, g_margin),
    'categorical_crossentropy': categorical_crossentropy
}

loss = [factory['categorical_crossentropy'], factory['triplet_loss']]
loss_weights = [1.0, 1.0]


''' optimizer '''
optimizer = Adam(g_base_lr)


''' callbacks '''
class EpochValCallback(keras.callbacks.Callback):
    # get model by self.model
    # get validation data by self.validation_data

    def __init__(self, v_datagen, loss):
        super().__init__()
        self.datagen = v_datagen
        self.loss_func = loss
        self.loss_history = []
        self.cmc_history = []
        self.mAP_history = []
        self.epoches = []

    def on_epoch_end(self, epoch, logs):
        print('learning rate: {}'.format(K.eval(self.model.optimizer.lr)))
        # the validation datagen yield the data in the format that
        # each four images/pids/cams belong to the same pid
        # imgs, pids, cams = self.datagen.flow()
        # feats = self.model.predict(imgs)[1] # TODO: now it is ft which should be fi
        # val_loss = K.eval(self.loss_func(None, feats))
        # print('val_loss: {}'.format(val_loss))
        # self.loss_history.append(val_loss)
        # dismat = K.eval(self.euclidean_distance(feats))
        # cmc, mAP = Evaluator._eval_func(dismat, pids, pids, cams, cams, 3)
        # print('val_cmc: {}, val_mAP: {}'.format(cmc, mAP))
        # self.cmc_history.append(cmc)
        # self.mAP_history.append(mAP)
        # self.epoches.append(epoch)

    # def on_train_end(self, logs):
    #     from matplotlib import pyplot as pl
    #     pl.ylim(0, int(self.loss_history[0] * 2))
    #     pl.xlim(0, len(self.epoches))
    #     pl.plot(self.epoches, self.loss_history)
    #     pl.plot(self.epoches, self.cmc_history)
    #     pl.plot(self.epoches, self.mAP_history)



def make_scheduler():
    def scheduler(epoch, lr):
        if epoch < 10:
            lr = g_base_lr * (epoch / 10)
        elif epoch == 10:
            lr = g_base_lr * 10
        elif epoch == 40:
            lr = g_base_lr
        elif epoch == 70:
            lr = g_base_lr * 0.1
        return lr
    return scheduler


check_point = kcb.ModelCheckpoint(
    './checkpoint/weights.{epoch:02d}-{loss:.2f}.h5',
    monitor='loss', save_weights_only=True,
    save_best_only=False, period=10
    )
tensor_board = kcb.TensorBoard(
    log_dir='./logs', batch_size=g_batch_size,
    write_graph=True, update_freq='epoch')

change_lr = kcb.LearningRateScheduler(make_scheduler())
# epochval = EpochValCallback(v_datagen, factory['triplet_loss'])

# callbacks = [change_lr, check_point, tensor_board, epochval]
callbacks = [change_lr, check_point, tensor_board]

''' metric '''
# TODO: add metric function
# def mymetric(y_true, y_pred):
#     pids, camids = y_true[0]
#     feature_maps = y_pred[1]
#     pass

''' model '''
g_input_shape = (g_img_h, g_img_w, g_img_ch)
base = ResNet50(include_top=False, weights='imagenet',
                input_tensor=Input(shape=g_input_shape),
                last_stride=g_stride)

feature_t = GlobalAveragePooling2D(name='GAP')(base.output)
feature_i = BatchNormalization(scale=False)(feature_t)
feat_model = Model(inputs=base.input, outputs=feature_i)
# feat_model = Model(inputs=base.input, outputs=feature_t)

prediction = Dense(g_num_classes, activation='softmax',
                   kernel_initializer='random_uniform',
                   name='FC', use_bias=False)(feature_i)
id_model = Model(inputs=base.input, outputs=prediction)
# id_model.summary()

model = Model(inputs=base.input, outputs=[prediction, feature_t])
# model.summary()

# for layer in model.layers:
#     layer.trainable = True

''' compile model '''
id_model.compile(optimizer=optimizer, loss=factory['categorical_crossentropy'],
                 metrics=['categorical_accuracy'])
model.compile(optimizer=optimizer, loss=loss,
              loss_weights=loss_weights, metrics=['categorical_accuracy'])

''' train '''
dummy = np.ones([g_batch_size, 2048])
train_datagen = TrainDataGenWrapper(t_datagen.flow, dummy, g_num_classes)


# #################################################################################

def train_only_id_loss(weight_path):
    print('[reid] only id loss training ...')
    id_model.fit_generator(
        train_datagen.flow(),
        steps_per_epoch=g_steps_per_epoch,
        epochs=g_epochs, verbose=1, callbacks=callbacks
    )
    id_model.save_weights(weight_path)

    print('[reid] benchmark ...')
    e = Evaluator(dataset, feat_model, g_img_h, g_img_w)
    e.compute()


def train_and_test(weight_path):
    print('[reid] training ...')
    model.fit_generator(
        train_datagen.flow(),
        steps_per_epoch=g_steps_per_epoch,
        epochs=g_epochs, verbose=1, callbacks=callbacks
    )
    model.save_weights(weight_path)

    print('[reid] benchmark ...')
    e = Evaluator(dataset, feat_model, g_img_h, g_img_w)
    e.compute()

# #################################################################################

# check list
# 1. learning rate
# 2. losses weights
# 3. model weight
# 4. epoch

train_and_test('weight_id_and_triplet_loss.h5')
train_only_id_loss('weights_only_id_loss.h5')
