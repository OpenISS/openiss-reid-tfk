from .market1501 import Market1501
from .cuhk03 import CUHK03
from .datagen import DataGen, TestDataGen
from .preprocess import to_categorical
from numpy import ones

dataset = None

class DataGenWrapper:
    def __init__(self, flow_func, dummy, num_classes):
        self.flow_func = flow_func
        self.dummy = dummy
        self.nc = num_classes

    def flow(self):
        while True:
            train_x, train_y = self.flow_func()
            train_y = to_categorical(train_y, self.nc)
            yield train_x, train_y
            # yield train_x, [train_y, self.dummy]
            # yield [train_x, train_y], [train_y, self.dummy]


def make_data_loader(cfg):
    global dataset
    # train_datagen, val_datagen, num_query, num_classes
    if cfg.DATASETS.NAME == 'market1501':
        dataset = Market1501(cfg.DATASETS.ROOT_DIR)

    elif cfg.DATASETS.NAME == 'cuhk03':
        dataset = CUHK03(cfg.DATASETS.ROOT_DIR)

    else:
        print('not done yet.')
        exit(-1)

    num_query = dataset.num_query_pids
    num_classes = dataset.num_train_pids

    bs = cfg.INPUT.IDS_PER_BATCH * cfg.INPUT.IMAGES_PER_ID
    dummy = ones([bs, 2048])
    # dummy = ones([bs, 1])

    datagen = DataGen(dataset.train, cfg)
    train_datagen = DataGenWrapper(datagen.flow, dummy, num_classes)

    # useless
    val_datagen = TestDataGen(dataset.query, cfg)
    # ------

    return train_datagen, val_datagen, num_query, num_classes


def get_dataset():
    global dataset
    return dataset
