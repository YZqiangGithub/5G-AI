import pandas as pd
import re
import collections
from itertools import chain
import os
from pandas import Series
import numpy as np
import sys
import traceback

class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # :
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数

def token_dict(new_key_param_dict):
    token_encode_dict = {}
    '''
    :param new_key_param_dict: 键为日志键，值为参数列表
    :return token_encode_dict: 格式为['dvsdd', [32,556,45,...],...]
    '''
    for key, value in new_key_param_dict.items():
        pass


def token_gene(logkey_param_dict):
    text = []
    new_key_param_dict = {}
    for key, value in logkey_param_dict.items():
        para1 = []

        for param in value:
            para2 = []
            for i in param:
                #文本过滤
                i = re.sub('=|\/|#|:|\[|\]|\'|\s+|\.|\-|\(|\)|,', '', str(i))
                text.append(i)
                para2.append(i)
            para1.append(para2)
        new_key_param_dict[key] = para1
    return new_key_param_dict

def get_param_dict(df_log):
    logkey_list = list(set([EventId for EventId in df_log['EventId']]))

    #从df_log中提取数据，初始化一些字典
    logkey_param_dict = {}
    logkey_content_dict = {}


    #字典初始化
    for key in logkey_list:
        logkey_param_dict[key] = []
        logkey_content_dict[key] = []


    #遍历df_log,将需要的数据以此添加到上述字典中
    for id in range(len(df_log)):
        log_key_tmp = df_log['EventId'][id]
        logkey_param_dict[log_key_tmp].append(df_log['ParamsList'][id])
        logkey_content_dict[log_key_tmp].append(df_log['ParamsList'][id])

    #对参数进行一些处理，去除一些符号
    new_key_param_dict = token_gene(logkey_param_dict)

    #建立一个字典，字典的键为日志键，值为一个字典（键为字符串，值为数字）
    token_encode_dict = token_dict(new_key_param_dict)


def params_value(df_train_log):
    key_param_dict_train, logkey_line_dict_train = get_param_dict(df_train_log)
