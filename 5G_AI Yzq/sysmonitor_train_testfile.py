import pandas as pd
import torch
from torch.nn import LSTM
import numpy as np
from torch.utils.data import DataLoader
from itertools import chain
from lib.common import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path):
    df = pd.read_csv(path)
    return df


if __name__ == '__main__':
    trainpath = "tmpdata/struct/sys_trian_filterStructured.csv"
    train_data = load_data(trainpath)
    paramslist = train_data['ParamsWithoutDigital'].values
    new_paramlist = []
    for param in paramslist:
        new_paramlist.append(eval(param))

    vocab = Vocab(new_paramlist)

    paramsdict_list = []
    for params in new_paramlist:
        params_dict = {}
        for idx, tokens in enumerate(params):
            params_dict[f'param{idx}'] = vocab[tokens]
        paramsdict_list.append(params_dict)

    print(paramsdict_list[:10])










