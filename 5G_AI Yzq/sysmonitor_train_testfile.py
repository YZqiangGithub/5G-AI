import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WORDS = 114
MAX_LEN = 14
EMB_SIZE = 128
HID_SIZE = 128
DROPOUT = 0.2
num_class = 2
epochs = 50
BATCH_SIZE = 256
lr = 0.001

class SysLogDataset(Dataset):
    def __init__(self, csc_file, transform = None):
        self.sysLogDigData = pd.read_csv(csc_file)
        self.transform = transform

    def __len__(self):
        return len(self.sysLogDigData)

    def __getitem__(self, idx):
        label = self.sysLogDigData.iloc[idx,0]
        log_data = self.sysLogDigData.iloc[idx,2:].values

        if self.transform:
            sample = self.transform(log_data)
        return log_data, label

class LogModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers = 2, bidirectional = True, dropout = 0.5):
        super(LogModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dp = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        x = self.dp(embedded)
        x, _ = self.lstm(x)
        x = self.dp(x)
        x = F.relu(self.fc1(x))
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        out = self.fc2(x)
        return out

if __name__ == '__main__':
    csv_path = 'tmpdata/struct/sys_train_digdata.csv'
    dataset = SysLogDataset(csv_path)
    trainloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=True, num_workers=1)


    logmodel = LogModel(vocab_size=MAX_WORDS, embedding_dim=MAX_LEN, hidden_dim=HID_SIZE, dropout=DROPOUT).to(device)
    print(f'train on {device}')
    optimizer = optim.Adam(logmodel.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = logmodel(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if(batch_idx + 1) % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(trainloader.dataset),
                           100. * batch_idx / len(trainloader), loss.item()))













