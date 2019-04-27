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
from data.datagen import DataGen, ValDataGen, TrainDataGenWrapper
from data.preprocess import imagenet_process
from backbone.resnet50 import ResNet50
from backbone.resnet50v2 import ResNet50V2
from tripletloss import triplet_loss
from evaluator import Evaluator
from logger import setup_logger

print('version of tensorflow: {}'.format(tf.VERSION))
print('version of keras: {}'.format(keras.__version__))


''' global variables '''
g_data_root  = '/home/h_lai/Documents/dl/reid/triplet/datasets'
# g_data_root = './datasets'
g_output_dir = './output'

g_resnet_version  = 'v1'
g_lr_warmup       = 'on'
g_random_erasing  = 'on'
g_label_smoothing = 'off'
g_net_last_stride = 1
g_bn_neck         = 'on'

g_num_ids  = 16
g_num_imgs = 4
g_img_h    = 256
g_img_w    = 128
g_img_ch   = 3

g_epochs   = 120
g_margin   = 0.3
g_base_lr  = 3.5e-5
g_stride   = 1
g_v_peroid = 40

g_dataset = Market1501(root=g_data_root)
t_datagen = DataGen(g_dataset.train, g_num_ids, g_num_imgs, g_img_w, g_img_h)
# v_datagen = ValDataGen(g_dataset.query, g_dataset.gallery, (g_img_h, g_img_w))

g_num_classes = g_dataset.num_train_pids
g_batch_size  = g_num_ids * g_num_imgs
g_steps_per_epoch = t_datagen.sampler.len // g_batch_size

g_dummy = np.ones([g_batch_size, 2048])
g_datagen = TrainDataGenWrapper(t_datagen.flow, g_dummy, g_num_classes)

g_train_logger = setup_logger('train', g_output_dir)
g_test_logger  = setup_logger('test', g_output_dir)


''' configuration validation '''



''' loss '''
# all possible loss function should register here
g_loss_factory = {
    'triplet_loss': triplet_loss(g_num_ids, g_num_imgs, g_margin),
    'categorical_crossentropy': categorical_crossentropy
}


g_loss = {
    'id': g_loss_factory['categorical_crossentropy'],
    'triplet': g_loss_factory['triplet_loss']
}
g_loss_weights = {'id': 1.0, 'triplet': 1.0}


''' optimizer '''
g_optimizer = Adam(g_base_lr)


''' model '''
tmp_shape = (g_img_h, g_img_w, g_img_ch)

if g_resnet_version == 'v1':
    g_base = ResNet50(include_top=False, weights='imagenet',
                    input_tensor=Input(shape=tmp_shape),
                    last_stride=g_stride)
elif g_resnet_version == 'v2':
    g_base = ResNet50V2(include_top=False, weights='imagenet',
                    input_tensor=Input(shape=tmp_shape),
                    last_stride=g_stride)

tmp_ft = GlobalAveragePooling2D(name='triplet')(g_base.output)
# tmp_fi = BatchNormalization(scale=False)(tmp_ft)
tmp_fi = BatchNormalization()(tmp_ft)
feat_model = Model(inputs=g_base.input, outputs=tmp_fi)
# feat_model = Model(inputs=base.input, outputs=feature_t)

tmp_pred = Dense(g_num_classes, activation='softmax',
                 kernel_initializer='random_uniform',
                 name='id', use_bias=False)(tmp_fi)

g_id_model = Model(inputs=g_base.input, outputs=tmp_pred)
g_model = Model(inputs=g_base.input, outputs=[tmp_pred, tmp_ft])

# g_id_model.summary()
# g_model.summary()


''' callbacks '''
class EpochValCallback(kcb.Callback):
    def __init__(self, evaluator, logger):
        super().__init__()
        self.evaluator = evaluator
        self.logger = logger

    def on_train_begin(self, logs=None):
        self.steps = self.params['steps']
        self.epochs = self.params['epochs']
        self.logger.info('training start')

    def on_epoch_end(self, epoch, logs):
        lr = K.eval(self.model.optimizer.lr)
        t_loss = logs.get('loss')
        t_acc = logs.get('id_categorical_accuracy')
        self.logger.info(
            '[{}/{}] ({}) epoch, loss: {}, acc: {}, lr: {}'
            .format(epoch + 1, self.epochs, self.steps, t_loss, t_acc, lr))
        if (epoch + 1) % g_v_peroid == 0:
            cmc, mAP = self.evaluator.compute(max_rank=5)
            self.logger.info('cmc: {}, mAP: {}'.format(cmc, mAP))

    def on_train_end(self, logs=None):
        self.logger.info('training finish')

def make_scheduler():
    '''
    Implementation of the warmup strategy of the learning rate.
    '''
    def scheduler(epoch, lr):
        if epoch < 10:
            lr = g_base_lr * ((epoch + 1) / 10)
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
    save_best_only=False, period=10)
tensor_board = kcb.TensorBoard(
    log_dir='./logs', batch_size=g_batch_size,
    write_graph=True, update_freq='epoch')

change_lr = kcb.LearningRateScheduler(make_scheduler())

g_tester = Evaluator(g_dataset, feat_model, g_img_h, g_img_w)
epochval = EpochValCallback(g_tester, g_train_logger)

g_callbacks = [change_lr, check_point, tensor_board, epochval]


''' compile model '''
g_id_model.compile(optimizer=g_optimizer,
                   loss=g_loss_factory['categorical_crossentropy'],
                   metrics=['categorical_accuracy'])
g_model.compile(optimizer=g_optimizer, loss=g_loss,
                loss_weights=g_loss_weights,
                metrics={'id': 'categorical_accuracy'})


# #################################################################################

def train_only_id_loss(weight_path):
    print('[reid] only id loss training ...')
    g_id_model.fit_generator(
        g_datagen.flow(),
        steps_per_epoch=g_steps_per_epoch,
        epochs=g_epochs, verbose=1, callbacks=g_callbacks
    )
    g_id_model.save_weights(weight_path)

    print('[reid] benchmark ...')
    e = Evaluator(g_dataset, feat_model, g_img_h, g_img_w)
    e.compute()


def train_and_test(weight_path):
    print('[reid] training ...')
    g_model.fit_generator(
        g_datagen.flow(),
        steps_per_epoch=g_steps_per_epoch,
        epochs=g_epochs, verbose=1, callbacks=g_callbacks
    )
    g_model.save_weights(weight_path)

    print('[reid] benchmark ...')
    e = Evaluator(g_dataset, feat_model, g_img_h, g_img_w)
    e.compute()

def train(target_model, save_weight_path):
    target_model.fit_generator(
        g_datagen.flow(),
        steps_per_epoch=g_steps_per_epoch,
        epochs=g_epochs, verbose=1, callbacks=g_callbacks
    )
    target_model.save_weights(save_weight_path)

def test(load_weight_path):
    print('[reid] benchmark ...')
    g_model.load_weights(load_weight_path)
    res = g_tester.compute()
    print(res)
# #################################################################################

'''
Training Checklist
    1. learning rate
    2. losses weights
    3. model weight
    4. epoch
    5. what kind of preprocessing
'''

# train_and_test('weight_id_and_triplet_loss.h5')
# train_only_id_loss('weights_only_id_loss.h5')
# test('backup/cmc85.39/weights.h5')
train(g_model, 'weights.h5')
