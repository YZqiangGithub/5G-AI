import pickle
from datetime import datetime
import collections

# 将对象以二进制形式保存
def save(filename, cls):
        with open(filename, 'wb') as f:
            pickle.dump(cls, f)


# 加载二进制形式的对象
def load(filename):
    with open(filename, 'rb') as f:
        cls = pickle.load(f)
        return cls


def time_elapsed(time_front, time_back, format="%H:%M:%S"):
    try:
        time_front_array = datetime.strptime(time_front, format)
        # print(time_front_array)
        time_back_array = datetime.strptime(time_back, format)
        #print(time_back_array)
        # time_elapsed = time_back_stamp - time_front_stamp
        time_elapsed = (time_back_array - time_front_array).seconds
        # if time_elapsed > 100:
        #     print(f"{time_front_array}  {time_back_array}  {time_elapsed}")
        return str(time_elapsed)
    except:
        return "0"

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