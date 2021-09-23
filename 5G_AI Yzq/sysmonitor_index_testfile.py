#coding:utf-8

import sys
from lib import spell
import re
import os

import pandas as pd
from lib.common import save, load, Vocab
import datetime
# from lib.common import time_elapsed

sys.setrecursionlimit(2000)

#获得时间戳
def UTC2Timestamp(t:str):
    begin = t.find('.')
    end = t.find('+')
    t = t.replace(t[begin:end],'')
    new_t = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    return int(new_t/300)

#生成分离日志信息的表达式
def gene_log_format_regex(log_format):
    headers = []  #csv文件头一行
    splitters = re.split(r'(<[^<>]+>)', log_format)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(r'\\ +', r'\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return  headers, regex

#将日志文件转换为DataFrame
def log2dataframe(log_file, headers, regex):
    log_messages= []
    with open(log_file, 'r') as fp:
        for line in fp.readlines():
            line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
            try:
                match = regex.search(line.strip())
                message = [UTC2Timestamp(match.group(header)) if header == 'TimeSlice' else match.group(header) for header in headers]
                log_messages.append(message)

            except Exception as e:
                pass

        logdf = pd.DataFrame(log_messages, columns=headers)
        return logdf


#从sysmonitor日志文件中分离出单纯的日志
def get_syscnt(log_file, log_format):
    headers, regex = gene_log_format_regex(log_format)
    df_log = log2dataframe(log_file, headers, regex)
    return df_log

def ParamsParse(txt):
    txt = re.sub(r'\[\d+\]', '[]', txt)
    txt = re.sub(r'(-[\d+]\(\d+\))?\(\d+\)|\d+\(\d+\)', '()', txt)
    txt = re.sub(r'\(\d+$', '(', txt)
    return txt

def spell_log(df_log, df_type='trian'):
    spell_result_path = 'tmpdata/SpellResult/spell.pkl'
    if os.path.isfile(spell_result_path):
        #加载保存好的结果
        slm = load(spell_result_path)
    else:
        #首先训练一边，找出所需日志键，保存到文件中
        #要选取能覆盖所有日志类型的数据来训练
        slm = spell.lcsmap('[\\s]+')
        for i in range(len(df_log)):
            log_message = df_log['Content'][i]
            sub = log_message.strip('\n')
            slm.insert(sub)
        save(spell_result_path, slm)

    #对每条日志进行训练一遍，保存到文件spell_result.txt中
    templates = [0] * df_log.shape[0]
    ids = [0] * df_log.shape[0]
    ParamsList = [0] * df_log.shape[0]
    Labels =  [0] * df_log.shape[0]
    # time_interval = [0] * df_log.shape[0]

    for i in range(len(df_log)):
        log_message = df_log['Content'][i].strip()
        obj = slm.insert(log_message)
        obj_json = obj.tojson(log_message)
        templates[i] = obj_json['lcsseq']
        ids[i] = obj_json['lcsseq_id']
        ParamsList[i] = obj_json['param']

    # for id in range(len(df_log)):
    #     if id == 0:
    #         time_interval[id] = '0'
    #     else:
    #         time_last = df_log['Time'][id-1]
    #         time_now  = df_log['Time'][id]
    #         elapsed = time_elapsed(time_last, time_now)
    #         time_interval[id] = elapsed
    #     ParamsList[id].append(time_interval[id])

    #将结果保存到df_log中
    # df_log['EventTemplate'] = templates #日志模板 日志键
    df_log['EventId'] = ids #事件向量
    df_log['ParamsList'] = ParamsList #参数列表

    # df_log.to_csv(f'tmpdata/struct/{df_type}_structured.csv', index = False)

    paramslist_withoutdig = []
    for Params in ParamsList:
        params = []
        for Param in Params:
            if len(Param) > 0:
                for p in Param:
                    param = ParamsParse(p)
                    params.append(param)
        paramslist_withoutdig.append(params)

    df_filter = pd.DataFrame()
    vocab = Vocab(paramslist_withoutdig) #构建词汇表
    max_idx = 0

    #获得各个参数列表的数字索引列表
    for params in paramslist_withoutdig:
        params_dict = {}
        for idx, tokens in enumerate(params):
            if max_idx < idx:
                max_idx = idx
            params_dict[f'param{idx}'] = vocab[tokens]
        df_filter = df_filter.append(params_dict, ignore_index=True)

    df_filter['TimeSlice'] = df_log['TimeSlice']
    df_filter['EventId'] = ids
    df_filter['Labels'] = Labels
    # df_filter['ParamsWithoutDigital'] = paramslist_withoutdig
    # df_filter.to_csv(f'tmpdata/struct/{df_type}_filterStructured.csv', index = False)

    col_seq = ['Labels', 'TimeSlice', 'EventId']
    col_seq.extend(['param'+str(idx) for idx in range(max_idx+1)]) #确定csv文件中各列的排列顺序
    df_filter.to_csv(f'./tmpdata/struct/{df_type}_digdata.csv', index=False, columns=col_seq, na_rep=0)

    return df_log



if __name__ == '__main__':
    syslog_format = '<TimeSlice>\|<Program>\|<Content>'

    train_sysmonitor_path = "./dataset/sysmoniter.txt"
    train_messages_path = "./dataset/messages.txt"

    tmpdata_path = ['struct', 'EventNpy', 'SpellResult', 'ParamsData', 'ParamsModel', 'ExecutePathModel']
    for path in tmpdata_path:
        if not os.path.exists(f'tmpdata/{path}'):
            os.makedirs(f'tmpdata/{path}')

    print('extract sysmonitor trian data')
    df_train_syslog = get_syscnt(train_sysmonitor_path, syslog_format)  #读取到的log
    df_train_syslog = spell_log(df_train_syslog, df_type='sys_trian')   #做生成经过日志模板提取， 参数提取之后的df_log


    # print('extract messages train data')
    # df_train_messlog = get_content()
    # df_train_messlog = spell_log()
