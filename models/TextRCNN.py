# -*- coding:utf-8 -*-
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'TextRCNN'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = None
        # 运行环境：Ascend, GPU, CPU
        self.device = 'GPU'

        self.dropout = 1.0
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 1
        self.batch_size = 128
        self.pad_size = 128
        self.learning_rate = 1e-3
        self.embedding_size = 300
        self.hidden_size = 256
        self.num_layers = 1


class Net(nn.Cell):
    def __init__(self, config):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(config.n_vocab, config.embedding_size, False)
        self.lstm = nn.LSTM(config.embedding_size, config.hidden_size,
                            config.num_layers,
                            bidirectional=True)
        # 由于mindspore不支持MaxPool1d，先凑合使用AvgPool1d
        self.maxpool = nn.AvgPool1d(config.pad_size)
        self.fc = nn.Dense(config.hidden_size * 2 + config.embedding_size,
                           config.num_classes)
        self.op = P.Concat(axis=2)
        self.relu = nn.ReLU()
        # 重新排列
        self.transpose = P.Transpose()
        self.squeeze = P.Squeeze()

        # size
        self.shape1 = Tensor(np.ones([2 * config.num_layers, config.batch_size, config.hidden_size]).astype(np.float32))

    def construct(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed, (self.shape1, self.shape1))
        out = self.op((embed, out))

        out = self.relu(out)
        # [128, 812, 128]
        out = self.transpose(out, (0, 2, 1))
        out = self.maxpool(out)
        # [128, 812, 1]
        out = self.squeeze(out)
        out = self.fc(out)

        return out