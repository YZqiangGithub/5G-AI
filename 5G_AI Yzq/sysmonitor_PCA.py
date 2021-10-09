#coding:utf-8
import pandas as pd
from collections import OrderedDict
import re
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
from lib.PCA import PCA
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

def load_logevent(logevent_file, whe_shuffle=True):
    logevent = pd.read_csv(logevent_file)
    x_train = logevent['EventSeq'].values

    if whe_shuffle:
        indexes = shuffle(np.arange(x_train.shape[0]))
        x_train = x_train[indexes]
    x_train = np.array([eval(x) for x in x_train])

    return x_train


if __name__ == '__main__':
    #1. 加载数据，处理特征向量
    syslog_trainfile = './tmpdata/struct/sys_train_evtseq.csv'
    msglog_trainfile = './tmpdata/struct/msg_train_evtseq.csv'
    syslog_testfile = './tmpdata/struct/sys_test_evtseq.csv'
    msglog_testfile = './tmpdata/struct/msg_test_evtseq.csv'

    sys_x_train = load_logevent(syslog_trainfile)
    msg_x_train = load_logevent(msglog_trainfile)

    sysfeature_extractor = FeatureExtractor()
    msgfeature_extractor = FeatureExtractor()

    sys_x_train = sysfeature_extractor.fit_transform(sys_x_train, term_weighting='tf-idf',normalization='zero-mean')
    msg_x_train = msgfeature_extractor.fit_transform(msg_x_train, term_weighting='tf-idf', normalization='zero-mean')

    sys_x_test  = load_logevent(syslog_testfile, whe_shuffle=False)
    msg_x_test  = load_logevent(msglog_testfile, whe_shuffle=False)
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
''
    #3. Use the trained model for anomaly detection
    sys_y_test = sys_model.predict(sys_x_test).astype(int)
    msg_y_test = msg_model.predict(msg_x_test).astype(int)

    # print(f'sys_test:{sys_y_test}')
    # print(f'msg_test:{msg_y_test}')

    label_res = np.hstack((msg_y_test, sys_y_test))
    logname_list = ['messages'] * msg_y_test.shape[0] + ['sysmonitor'] * sys_y_test.shape[0]
    sys_timeslice = pd.read_csv(syslog_testfile)['TimeSlice'].values
    msg_timeslice = pd.read_csv(msglog_testfile)['TimeSlice'].values

    timeslice_res = np.hstack((msg_timeslice, sys_timeslice))

    preRes_dict = {}
    preRes_dict['TimeSlice'] = timeslice_res
    preRes_dict['LogName'] = logname_list
    preRes_dict['Label'] = label_res

    pred_res = pd.DataFrame(preRes_dict)
    pred_res.to_csv('./predict_result.csv', index=False)

