# coding: UTF-8

import os
import pickle
import time
from tqdm import tqdm
from datetime import timedelta

import numpy as np
from mindspore import dataset as ds

# 词表长度限制(主要控制中文单字的大小）
MAX_VOCAB_SIZE = 10000
# 定义填充未知词和补全词
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    构造词汇表
    :param file_path:
    :param tokenizer: 分割方式
    :param max_size:
    :param min_freq:
    :return:
    """
    vocab_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue

            # Tab为分隔符
            content = line.split('\t')[0]
            for word in tokenizer(content):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1

        vocab_list = sorted(
            [item for item in vocab_dict.items() if item[1] >= min_freq],
            key=lambda x: x[1], reverse=True)[:max_size]

        vocab_dict = {
            word_count[0]: idx for idx, word_count in enumerate(vocab_list)
        }
        vocab_dict.update({
            UNK: len(vocab_dict),
            PAD: len(vocab_dict) + 1
        })

    return vocab_dict


def build_dataset(config, use_word=False):
    """
    构造数据集
    :param config:
    :param use_word: 决定是否依据词还是字符进行分词
    :return:
    """
    if use_word:
        tokenizer = lambda x: x.split(' ')
    else:
        tokenizer = lambda x: [y for y in x]

    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer,
                            max_size=MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(vocab, open(config.vocab_path, 'wb'))
    print(f'Vocab size: {len(vocab)}')

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    validation = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, validation, test


class IterDatasetGenerator(object):
    def __init__(self, datas):
        self.__index = 0
        self.__data = datas

    def __next__(self):
        if self.__index >= len(self.__data):
            raise StopIteration

        x, y = self.__data[self.__index][0:2]
        self.__index += 1
        x = np.array(x)
        y = np.array([y])
        return (x, y)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.__data)


def build_iterator(dataset):
    dataset_generator = IterDatasetGenerator(dataset)
    data_iter = ds.GeneratorDataset(
        dataset_generator, ['data', 'label'], shuffle=False)
    return data_iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_data(config):
    start = time.time()
    vocab, train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data)
    dev_iter = build_iterator(dev_data)
    test_iter = build_iterator(test_data)
    print('Time usage:{}s'.format(time.time() - start))
    return vocab, train_iter, dev_iter, test_iter
