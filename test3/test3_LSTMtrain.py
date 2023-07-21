import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import utils

import torchtext
from tqdm import tqdm
from torchtext.datasets import IMDB

from torchtext.datasets.imdb import NUM_LINES
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
import matplotlib.pyplot as plt # plt 用于显示图片
plt.switch_backend('agg')

# LSTM set
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,bidirectional=True,dropout=0)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = F.avg_pool2d(out,(out.shape[1],1)).squeeze()
        out = self.fc(out)

        return out


# 构建IMDB Dataloader
BATCH_SIZE = 32

def yeild_tokens(train_data_iter, tokenizer):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        yield tokenizer(comment)  # 字符串转换为token索引的列表
        
train_data_iter = IMDB(root="/home/crq2/AI/dataset", split="train")  # Dataset类型的对象
tokenizer = get_tokenizer("basic_english")
# 只使用出现次数大约20的token
vocab = build_vocab_from_iterator(yeild_tokens(train_data_iter, tokenizer), min_freq=20, specials=["<unk>"])
vocab.set_default_index(0)  # 特殊索引设置为0
print(f'单词表大小: {len(vocab)}')

def collate_fn(batch):
    """
    对DataLoader所生成的batch进行处理
    """
    target = []
    token_index = []
    max_length = 0  # 最大的token长度
    for i, (label, comment) in enumerate(batch):
        tokens = tokenizer(comment)
        token_index.append(vocab(tokens))  # 字符列表转换为索引列表
        
        # 确定最大的句子长度
        if len(tokens) > max_length:
            max_length = len(tokens)
        # label=2 means pos,else means neg
        if label == 1:
            target.append(0)
        else:
            target.append(1)

    token_index = [index + [0]*(max_length-len(index)) for index in token_index]
    # one-hot接收长整形的数据，所以要转换为int64
    return (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32))


train_data_iter = IMDB(root="/home/crq2/AI/dataset", split="train")  # Dataset类型的对象
train_data_loader = torch.utils.data.DataLoader(
    to_map_style_dataset(train_data_iter), batch_size=BATCH_SIZE,collate_fn=collate_fn,shuffle=True)
# 数据检查
for idx,(label,text) in enumerate(train_data_loader):
    print("idx:",idx)
    print("lable:",label)
    print("text:",text)
    break

test_data_iter = IMDB(root="/home/crq2/AI/dataset", split="test")  # Dataset类型的对象
# collate校对
test_data_loader = torch.utils.data.DataLoader(
    to_map_style_dataset(test_data_iter), batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle=True)

input_size = len(vocab)
hidden_size = 128
num_layers = 2
output_size = 2

model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    corgt = 0

    for batch_index, (target, token_index) in enumerate(iterator):
        optimizer.zero_grad()
        token_index = token_index.to(device)
        target = target.to(device)
        logits = model(token_index)
        # one-hot需要转换float32才可以训练
        #target_onehot = F.one_hot(target, num_classes=2).to(torch.float32)
        loss = criterion(logits, target)
        pred = logits.max(-1,keepdim=True)[1]
        corgt += (pred == target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        #if batch_index % 10==0:
            #print('Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_index * len(token_index), len(iterator.dataset),100. * batch_index / len(iterator), loss.item()))
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), corgt / len(iterator.dataset)

def evaluate_every_class(model, iterator):
    model.eval()
    corgt = 0
    pos_corgt = 0
    neg_corgt = 0
    pos_total = 0
    neg_total = 0

    for batch_index, (target, token_index) in enumerate(iterator):
        token_index = token_index.to(device)
        target = target.to(device)
        logits = model(token_index)
        # one-hot需要转换float32才可以训练
        #target_onehot = F.one_hot(target, num_classes=2).to(torch.float32)
        pred = logits.max(-1,keepdim=True)[1]
        for i in range(target.shape[0]):
            gt = target.view_as(pred)[i]
            pre = pred[i]
            if gt == 0:
                neg_total+=1
                if gt == pre:
                    corgt+=1
                    neg_corgt+=1
            else:
                pos_total+=1
                if gt == pre:
                    corgt+=1
                    pos_corgt+=1

    return  corgt / len(iterator.dataset),pos_corgt/pos_total,neg_corgt/neg_total


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    corgt = 0

    for batch_index, (target, token_index) in enumerate(iterator):
        token_index = token_index.to(device)
        target = target.to(device)
        logits = model(token_index)
        # one-hot需要转换float32才可以训练
        #target_onehot = F.one_hot(target, num_classes=2).to(torch.float32)
        loss = criterion(logits, target)
        pred = logits.max(-1,keepdim=True)[1]
        corgt += (pred == target.view_as(pred)).sum().item()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), corgt / len(iterator.dataset)

N_EPOCHS = 10



best_valid_loss = float('inf')
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_data_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_data_loader, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'test3/lstm_model.pt')

    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_loss_history.append(valid_loss)
    test_acc_history.append(valid_acc)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(N_EPOCHS), train_loss_history,test_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend(labels=["train","test"],loc="best")

plt.subplot(1, 2, 2)
plt.plot(range(N_EPOCHS), train_acc_history,test_acc_history)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend(labels=["train","test"],loc="best")
plt.show()
curve_fig = "test3/lstm_curve.png"
plt.savefig(curve_fig)


acc, pos_acc,neg_acc= evaluate_every_class(model, test_data_loader)
print(f'LSTM:\nacc: {acc*100:.2f}% \npos_acc: {pos_acc*100:.2f}% \nneg_acc: {neg_acc*100:.2f}% \n')