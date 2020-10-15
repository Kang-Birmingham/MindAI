# -*- coding:utf-8 -*-
from importlib import import_module

from util import load_data
from train_eval import train, test
from mindspore import context


def main():
    # 数据集名称
    dataset = 'THUCNews'
    # 模型名称
    model_name = 'TextRCNN'

    # 读取模型
    model = import_module('models.' + model_name)

    # 获取模型初始配置
    config = model.Config(dataset)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device)


    # 获取词汇表和数据
    vocab, train_iter, dev_iter, test_iter = load_data(config)

    # 设置词汇表的大小
    config.n_vocab = len(vocab)

    # 模型训练
    network = model.Net(config)
    trained_model = train(config, network, train_iter)

    # 模型测试
    test(trained_model, test_iter)


if __name__ == '__main__':
    main()
