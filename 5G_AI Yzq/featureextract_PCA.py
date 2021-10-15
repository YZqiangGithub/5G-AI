# coding:utf-8

import sys
from lib import spell
import re
import os

import pandas as pd
from lib.common import save, load, Vocab
import datetime
from collections import OrderedDict
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
from lib.PCA import PCA

# from lib.common import time_elapsed

sys.setrecursionlimit(2000)


# 获得时间戳
def UTC2Timestamp(t: str):
    begin = t.find('.')
    end = t.find('+')
    t = t.replace(t[begin:end], '')
    new_t = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    return int(new_t / 300)


# 生成分离日志信息的表达式
def gene_log_format_regex(sysmonitor, log_format):
    headers = set()  # csv文件头一行
    regexes = []  # 针对日志的不同格式的匹配列表
    if sysmonitor:
        splitters = re.split(r'(<[^<>]+>)', log_format[0])
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r'\\ +', r'\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.add(header)
        regex = re.compile('^' + regex + '$')
        regexes.append(regex)
    else:
        for format in log_format:
            splitters = re.split(r'(<[^<>]+>)', format)
            regex = ''
            for k in range(len(splitters)):
                if k % 2 == 0:
                    splitter = re.sub(r'\\ +', r'\\s+', splitters[k])
                    regex += splitter
                else:
                    header = splitters[k].strip('<').strip('>')
                    regex += '(?P<%s>.*?)' % header
                    headers.add(header)
            regex = re.compile('^' + regex + '$')
            regexes.append(regex)

    return headers, regexes


# 将日志文件转换为DataFrame
def log2dataframe(log_file, headers, regexes, sysmonitor):
    log_messages = []
    with open(log_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)

            for regex in regexes:
                try:
                    match = regex.search(line.strip())
                    if match is None:
                        continue
                    message = [
                        UTC2Timestamp(match.group(header)) if header == 'TimeSlice' else match.group(header) for header in
                        headers]
                    log_messages.append(message)

                except Exception as e:
                    pass

        logdf = pd.DataFrame(log_messages, columns=headers)
        return logdf


# 从日志文件中分离出单纯的日志
def get_content(log_file, sysmonitor, *log_format):
    headers, regexes = gene_log_format_regex(sysmonitor, log_format)
    df_log = log2dataframe(log_file, headers, regexes, sysmonitor)
    return df_log


# def ParamsParse(txt, sysmonitor, idx):
#     if sysmonitor:
#         txt = re.sub(r'\[\d+\]', '[]', txt)
#         txt = re.sub(r'(-[\d+]\(\d+\))?\(\d+\)|\d+\(\d+\)', '()', txt)
#         txt = re.sub(r'\(\d+$', '(', txt)
#     else:
#         if idx != 0 and txt == '0':
#             txt = ''
#         elif txt == '0:0':
#             txt = '0.0.0.0:0'
#         else:
#             txt = re.sub(r'^\d+$|\d+_\d+\+\d+_\d+|\d+-\d+-\d+_\d+-\d+-\d+', r'*', txt)
#             txt = re.sub(r'\[\d+\]', '[*]', txt)
#             txt = re.sub(r'\(\d+\)', '(*)', txt)
#             txt = re.sub(r'\[\d+', '\[*', txt)
#             txt = re.sub(r'\d+\]', '*]', txt)
#             txt = re.sub(r'=\d+', '=*', txt)
#     return txt


def spell_log(df_log, df_type='trian', sysmonitor=True):
    spell_result_path = 'tmpdata/SpellResult/sysspell.pkl' if sysmonitor else './tmpdata/SpellResult/msgspell.pkl'
    if os.path.isfile(spell_result_path):
        # 加载保存好的结果
        slm = load(spell_result_path)
    else:
        # 首先训练一边，找出所需日志键，保存到文件中
        # 要选取能覆盖所有日志类型的数据来训练
        slm = spell.lcsmap(r'[\s|\||=]+')
        for i in range(len(df_log)):
            log_message = df_log['Content'][i]
            sub = log_message.strip('\n')
            sub = re.sub('\[\d+\]', '[~]', sub)
            sub = re.sub('\(\d+\)', '(~)', sub)
            slm.insert(sub)
        save(spell_result_path, slm)

    # 对每条日志进行训练一遍，保存到文件中
    templates = [0] * df_log.shape[0]
    ids = [0] * df_log.shape[0]
    ParamsList = [0] * df_log.shape[0]
    # Labels = [0] * df_log.shape[0]

    for i in range(len(df_log)):
        log_message = df_log['Content'][i].strip()
        log_message = re.sub('\[\d+\]', '[~]', log_message)
        log_message = re.sub('\(\d+\)', '(~)', log_message)
        obj = slm.insert(log_message)
        obj_json = obj.tojson(log_message)
        templates[i] = obj_json['lcsseq']
        ids[i] = obj_json['lcsseq_id']
        ParamsList[i] = obj_json['param']

    # 将结果保存到df_log中
    df_log['EventId'] = ids  # 事件向量
    df_log['ParamsList'] = ParamsList #参数列表
    df_log['EventTemplate'] = templates  # 日志模板 日志键
    df_log.to_csv(f'tmpdata/struct/{df_type}_structured.csv', index = False)

    #以数组存储事件序列
    event_dict = OrderedDict()
    for idx, timetemp in enumerate(df_log['TimeSlice']):
        timetemp = str(timetemp)
        if not timetemp in event_dict.keys():
            event_dict[timetemp] = []
        event_dict[timetemp].append('E' + str(ids[idx]))

    df_evtseq = pd.DataFrame(list(event_dict.items()), columns=['TimeSlice', 'EventSeq'])

    df_evtseq.to_csv(f'./tmpdata/struct/{df_type}_evtseq.csv', index=False)

    return df_evtseq

    ##分割事件序列按列存入文件
    # df_eventid = pd.DataFrame()
    # max_idx = 0
    # current_temp = 0
    # event_dict = {}
    # for idx, timetemp in enumerate(df_log['TimeSlice']):
    #     if current_temp != timetemp:
    #         if len(event_dict) != 0:
    #             df_eventid = df_eventid.append(event_dict ,ignore_index=True)
    #         current_temp = timetemp
    #         max_idx = len(event_dict) - 1 if len(event_dict) - 1 > max_idx else max_idx
    #         event_dict.clear()
    #         current_eventidx = 0
    #
    #         event_dict['TimeSlice'] = current_temp
    #         event_dict[f'evtid{current_eventidx}'] = ids[idx]
    #         current_eventidx += 1
    #     else:
    #         event_dict[f'evtid{current_eventidx}'] = ids[idx]
    #         current_eventidx += 1
    #
    # Labels = [0] * df_eventid.shape[0]
    # df_eventid['Labels'] = Labels
    #
    # col_seq = ['Labels', 'TimeSlice']
    # col_seq.extend(['evtid'+str(idx) for idx in range(max_idx)]) #确定csv文件中各列的排列顺序
    # df_eventid= df_eventid.fillna(0)
    # df_eventid = df_eventid.astype(int)
    # df_eventid.to_csv(f'./tmpdata/struct/{df_type}_evtiddata.csv', index=False, columns=col_seq)

    # paramslist_withoutdig = []
    # for Params in ParamsList:
    #     params = []
    #     for Param in Params:
    #         if len(Param) > 0:
    #             for idx, p in enumerate(Param):
    #                 param = ParamsParse(p, sysmonitor, idx)
    #                 params.append(param)
    #     paramslist_withoutdig.append(params)
    #
    # df_filter = pd.DataFrame()
    #
    # vocab = Vocab(paramslist_withoutdig) #构建词汇表
    # max_idx = 0
    # #获得各个参数列表的数字索引列表
    # for params in paramslist_withoutdig:
    #     params_dict = {}
    #     for idx, tokens in enumerate(params):
    #         if max_idx < idx:
    #             max_idx = idx
    #         params_dict[f'param{idx}'] = int(vocab[tokens])
    #     df_filter = df_filter.append(params_dict, ignore_index=True)
    #
    # df_filter['TimeSlice'] = df_log['TimeSlice']
    # df_filter['EventId'] = ids
    # df_filter['Labels'] = Labels
    #
    # df_filter['ParamsWithoutDigital'] = paramslist_withoutdig
    # df_filter.to_csv(f'tmpdata/struct/{df_type}_filterStructured.csv', index=False)
    #
    # col_seq = ['Labels', 'TimeSlice', 'EventId']
    # col_seq.extend(['param'+str(idx) for idx in range(max_idx+1)]) #确定csv文件中各列的排列顺序
    # df_filter = df_filter.fillna(0)
    # df_filter = df_filter.astype(int)
    # df_filter.to_csv(f'./tmpdata/struct/{df_type}_digdata.csv', index=False, columns=col_seq)

class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None
        self.oov = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1):
        """ Fit and transform the data matrix

        Arguments
        ---------
            X_seq: ndarray, log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`
            oov: bool, whether to use OOV event
            min_count: int, the minimal occurrence of events (default 0), only valid when oov=True.

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')
        self.term_weighting = term_weighting  # tf-idf
        self.normalization = normalization  # zero-mean
        self.oov = oov  # false

        X_counts = []
        for i in range(X_seq.shape[0]):
            # 将每个日志的每个事件计数组成event_counts
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        # print(X_seq)
        # [list(['E9']) list(['E5', 'E5']) list(['E5', 'E22', 'E5', 'E5', 'E11'])]
        # print(X_counts)
        # [Counter({'E9': 1}), Counter({'E5': 2}), Counter({'E5': 3, 'E22': 1, 'E11': 1})]
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)  # 填充空值
        # print(X_df)
        #        E9   E5  E22  E11
        #     0  1.0  0.0  0.0  0.0
        #     1  0.0  2.0  0.0  0.0
        #     2  0.0  3.0  1.0  1.0

        self.events = X_df.columns  # 所有的event事件
        X = X_df.values  # 上面的矩阵 3行四列

        num_instance, num_event = X.shape  # 行数 列数 3，4

        # TF-IDF模型  Term Frequency Inverse Document Frequency
        # TF-IDF用以评估一字词对于一个语料库中的其中一份文件的重要程度
        # http://www.ruanyifeng.com/blog/2013/03/tf-idf.html
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0)  # 列相加
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        # 经过pf-idf模型
        # [[2. 0. 0. 0.]        [[ 0.13515503 -0.36620409 -0.36620409 -0.36620409]
        # [0. 1. 0. 0.]    -->   [-0.67577517  0.73240819 -0.36620409 -0.36620409]
        # [3. 0. 1. 1.]]        [ 0.54062014 -0.36620409  0.73240819  0.73240819]]

        # 标准化，数据符合标准正态分布，即均值为0，标准差为1
        # print(X)
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        # [[0.81093021 0.         0.         0.        ]
        #  [0.         1.09861228 0.         0.        ]
        #  [1.21639531 0.         1.09861228 1.09861228]]
        # -->
        # [[ 0.13515503 -0.36620409 -0.36620409 -0.36620409]
        #  [-0.67577517  0.73240819 -0.36620409 -0.36620409]
        #  [ 0.54062014 -0.36620409  0.73240819  0.73240819]]
        X_new = X

        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new

    def transform(self, X_seq):
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values
        if self.oov:
            oov_vec = np.sum(X_df[X_df.columns.difference(self.events)].values > 0, axis=1)
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])

        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))

        return X_new


if __name__ == '__main__':
    syslog_format = '<TimeSlice>\|<Content>'
    message_format_v1 = '<TimeSlice>\|<Content>'
    message_format_v2 = '<TimeSlice> <Content>'

    train_sysmonitor_path = "./dataset/sysmoniter.txt"
    train_messages_path = "./dataset/messages.txt"
    test_sysmonitor_path = './dataset/sysmonitor_testset.txt'
    test_messages_path = './dataset/messages_testset.txt'

    tmpdata_path = ['struct', 'SpellResult']
    for path in tmpdata_path:
        if not os.path.exists(f'tmpdata/{path}'):
            os.makedirs(f'tmpdata/{path}')

    print('extract sysmonitor train data')
    df_train_syslog = get_content(train_sysmonitor_path, True, syslog_format)  #读取到的log
    sys_x_train = spell_log(df_train_syslog, df_type='sys_train', sysmonitor=True)   #做生成经过日志模板提取， 参数提取之后的df_log
    sys_x_train = sys_x_train['EventSeq'].values
    sys_x_train = shuffle(sys_x_train)

    print('extract messages train data')
    df_train_msglog = get_content(train_messages_path, False, message_format_v1, message_format_v2)
    msg_x_train = spell_log(df_train_msglog, df_type='msg_train', sysmonitor=False)
    msg_x_train = msg_x_train['EventSeq'].values
    msg_x_train = shuffle(msg_x_train)

    print('extract sysmonitor test data')
    df_test_syslog = get_content(test_sysmonitor_path, True, syslog_format)  #读取到的log
    sys_x_test = spell_log(df_test_syslog, df_type='sys_test', sysmonitor=True)   #做生成经过日志模板提取， 参数提取之后的df_log

    print('extract messages test data')
    df_test_msglog = get_content(test_messages_path, False, message_format_v1, message_format_v2)
    msg_x_test = spell_log(df_test_msglog, df_type='msg_test', sysmonitor=False)

    sys_timeslice = sys_x_test['TimeSlice'].values
    msg_timeslice = msg_x_test['TimeSlice'].values

    timeslice_res = np.hstack((msg_timeslice, sys_timeslice))

    sys_x_test = sys_x_test['EventSeq'].values
    msg_x_test = msg_x_test['EventSeq'].values

    sysfeature_extractor = FeatureExtractor()
    msgfeature_extractor = FeatureExtractor()

    sys_x_train = sysfeature_extractor.fit_transform(sys_x_train, term_weighting='tf-idf',normalization='zero-mean')
    msg_x_train = msgfeature_extractor.fit_transform(msg_x_train, term_weighting='tf-idf', normalization='zero-mean')

    msg_x_test = msgfeature_extractor.transform(msg_x_test)
    sys_x_test = sysfeature_extractor.transform(sys_x_test)

    ## 2. Train an unsupervised model
    print('Train phase:')
    # Initialize PCA, or other unsupervised models, LogClustering, InvariantsMiner
    sys_model = PCA()
    msg_model = PCA()
    # Model hyper-parameters may be sensitive to log data, here we use the default for demo
    sys_model.fit(sys_x_train)
    msg_model.fit(msg_x_train)

    #3. Use the trained model for anomaly detection
    sys_y_test = sys_model.predict(sys_x_test).astype(int)
    msg_y_test = msg_model.predict(msg_x_test).astype(int)

    # print(f'sys_test:{sys_y_test}')
    # print(f'msg_test:{msg_y_test}')

    label_res = np.hstack((msg_y_test, sys_y_test))
    logname_list = ['messages'] * msg_y_test.shape[0] + ['sysmonitor'] * sys_y_test.shape[0]


    preRes_dict = {}
    preRes_dict['TimeSlice'] = timeslice_res
    preRes_dict['LogName'] = logname_list
    preRes_dict['Label'] = label_res

    pred_res = pd.DataFrame(preRes_dict)
    pred_res.to_csv('./predict_result.csv', index=False)
