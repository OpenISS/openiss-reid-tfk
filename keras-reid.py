
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
# print('[reid] enable eager execution: {}'.format(tf.executing_eagerly()))
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.resnet50 import ResNet50
from keras import callbacks as kcb
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from data.market1501 import Market1501
from data.datagen import DataGen
from data.preprocess import imagenet_process

print('version of tensorflow: {}'.format(tf.VERSION))
print('version of keras: {}'.format(keras.__version__))


''' constant '''
g_data_root = '/home/h_lai/Documents/dl/reid/triplet/datasets'

g_num_ids = 16
g_num_imgs = 4
g_img_h = 256
g_img_w = 128
g_img_ch = 3

g_epoch = 500

''' dataset '''
class DataGenWrapper:
    def __init__(self, flow_func, dummy, num_classes):
        self.flow_func = flow_func
        self.dummy = dummy
        self.nc = num_classes

    def flow(self):
        while True:
            train_x, train_y = self.flow_func()
            train_y = to_categorical(train_y, self.nc)
            yield train_x, [train_y, self.dummy]


dataset = Market1501(root=g_data_root)
datagen = DataGen(dataset.train, g_num_ids, g_num_imgs, g_img_w, g_img_h)
datagen1 = DataGen(dataset.query, g_num_ids, g_num_imgs, g_img_w, g_img_h)

g_num_classes = dataset.num_train_pids
g_batch_size = g_num_ids * g_num_imgs
g_steps_per_epoch = datagen.sampler.len // g_batch_size

''' model '''
g_input_shape = (g_img_h, g_img_w, g_img_ch)
base = ResNet50(include_top=False, weights='imagenet',
                input_tensor=Input(shape=g_input_shape))

feature = GlobalAveragePooling2D(name='GAP')(base.output)
feat_model = Model(inputs=base.input, outputs=feature)

prediction = Dense(g_num_classes, activation='softmax', name='FC')(feature)
id_model = Model(inputs=base.input, outputs=prediction)

model = Model(inputs=base.input, outputs=[prediction, feature])
# model.summary()

''' loss '''
from tripletloss import triplet_hard_loss
factory = {
    'triplet_hard': triplet_hard_loss(g_num_ids, g_num_imgs, 0.3),
    'categorical_crossentropy': tf.keras.losses.categorical_crossentropy
}

# loss = ['categorical_crossentropy']
loss = [factory['categorical_crossentropy'], factory['triplet_hard']]
loss_weights = [1.0, 1.0]

''' optimizer '''
g_base_lr = 3.5e-5
optimizer = tf.keras.optimizers.Adam(g_base_lr)

''' callbacks '''
def make_scheduler():
    def scheduler(epoch, lr):
        if epoch < 10:
            lr = g_base_lr * (epoch / 10)
        elif epoch == 10:
            lr = g_base_lr * 0.1
        elif epoch == 40:
            lr = g_base_lr
        elif epoch == 70:
            lr = g_base_lr * 10
        return lr
    return scheduler


check_point = kcb.ModelCheckpoint(
    './checkpoint/weights.{epoch:02d}-{loss:.2f}.h5',
    monitor='val_loss', save_weights_only=True,
    save_best_only=False, period=10
    )
tensor_board = kcb.TensorBoard(
    log_dir='./logs', batch_size=g_batch_size, write_graph=True)

change_lr = kcb.LearningRateScheduler(make_scheduler())
callbacks = [change_lr, check_point, tensor_board]

''' compile model '''
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

''' train '''
dummy = np.ones([g_batch_size, 2048])
train_datagen = DataGenWrapper(datagen.flow, dummy, g_num_classes)

val_datagen = ImageDataGenerator(preprocessing_function=imagenet_process)

def train():
    print('[reid] training ...')
    model.fit_generator(
        train_datagen.flow(),
        steps_per_epoch=g_steps_per_epoch,
        epochs=g_epoch, verbose=1, callbacks=callbacks
        )
    model.save_weights('weights.h5')

def test():
    print('[reid] benchmark ...')
    from eval import Evaluator
    # model.load_weights('weights.h5')
    # model.load_weights('checkpoint/weights.260-0.13.h5')
    model.load_weights('/home/h_lai/Documents/dl/myreid/checkpoint/weights.500-1.42.h5')
    e = Evaluator(dataset, feat_model, g_img_h, g_img_w)
    e.compute()

# train()
test()
