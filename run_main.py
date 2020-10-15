# -*- coding:utf-8 -*-
from importlib import import_module

from util import load_data
from model_train import train

def main():
    # 数据集名称
    dataset = 'THUCNews'
    # 模型名称
    model_name = 'TextRCNN'
    
    # 读取模型
    model = import_module('models.' + model_name)

    # 获取模型初始配置
    config = model.Config(dataset)

    # 获取词汇表和数据
    vocab, train_iter, dev_iter, test_iter = load_data(config)

    # 设置词汇表的大小
    config.n_vocab = len(vocab)

    # 模型训练
    network = model.Net(config)
    train(config, network, train_iter)


if __name__ == '__main__':
    main()
