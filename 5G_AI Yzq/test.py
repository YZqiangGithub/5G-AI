import pandas as pd


trainpath = 'tmpdata/struct/sys_train_digdata.csv'
SysLogDigData = pd.read_csv(trainpath)

print(SysLogDigData.iloc[0,2:].values)
