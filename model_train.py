# -*- coding:utf-8 -*-
import time

import numpy as np
from sklearn import metrics
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits


def train(config, net, train_iter):
    """
    模型训练
    :param config:
    :param net:
    :param train_iter:
    :return:
    """
    print("============== Starting Training ==============")

    # 定义损失函数
    net_loss = SoftmaxCrossEntropyWithLogits()
    # 定义优化器
    net_opt = nn.Adam(net.trainable_params(),
                      learning_rate=config.learning_rate)

    model = Model(net, net_loss, net_opt, metrics={'Accuracy': Accuracy()})
    model.train(config.num_epochs, train_iter)
