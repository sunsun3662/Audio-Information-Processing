# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== LSTM 实现 =====
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.W_x.data.uniform_(-stdv, stdv)
        self.W_h.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, x, state):
        h, c = state
        gates = x @ self.W_x + h @ self.W_h + self.bias
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)
        c_t = f_t * c + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(MultiLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(layer_input_size, hidden_size))

    def forward(self, x, states=None):
        seq_len, batch_size, _ = x.size()
        if states is None:
            h = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
            c = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h, c = states
        # 避免inplace操作：每次重新构建h和c
        h_list = [h[i] for i in range(self.num_layers)]
        c_list = [c[i] for i in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            inp = x[t]
            for layer in range(self.num_layers):
                h_new, c_new = self.cells[layer](inp, (h_list[layer], c_list[layer]))
                h_list[layer] = h_new
                c_list[layer] = c_new
                inp = h_new
            outputs.append(h_list[-1])
        output = torch.stack(outputs, dim=0)
        h_out = torch.stack(h_list, dim=0)
        c_out = torch.stack(c_list, dim=0)
        return output, (h_out, c_out)


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_num):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 使用PyTorch内置LSTM（cuDNN加速）
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layer_num)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, hidden)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden


# ===== 训练函数 =====
def validate(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_ in val_loader:
            data_ = data_.long().transpose(1, 0).contiguous().to(device)
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train(train_loader, val_loader, model, optimizer, criterion, device,
          epochs=10, save_interval=5, model_prefix='checkpoints/tang', model_path='checkpoints/tang_model.pth'):
    loss_list = []
    val_loss_list = []
    model.to(device)
    os.makedirs(model_prefix, exist_ok=True)

    print("Training started...")
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for ii, data_ in enumerate(progress_bar):
            data_ = data_.long().transpose(1, 0).contiguous().to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] train loss: {avg_loss:.4f}")

        val_loss = validate(val_loader, model, criterion, device)
        val_loss_list.append(val_loss)
        print(f"Epoch [{epoch+1}/{epochs}] validation loss: {val_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            model_path_epoch = os.path.join(model_prefix, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_path_epoch)
            print(f"The model has been saved to {model_path_epoch}")

    torch.save(model.state_dict(), model_path)
    print(f"The final model has been saved to {model_path}")
    return loss_list, val_loss_list


# ===== 主程序 =====
if __name__ == '__main__':
    # 超参数
    layer_num = 2
    embedding_dim = 256
    hidden_dim = 256
    lr = 1e-3
    epochs = 50
    batch_size = 128
    save_interval = 10
    model_path = './checkpoints/tang_model.pth'
    model_prefix = './checkpoints/tang'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    pickle_path = "./data/tang.npz"
    datas = np.load(pickle_path, allow_pickle=True)
    data = datas.get('data')
    word2ix = datas['word2ix'].item()
    ix2word = datas['ix2word'].item()
    vocab_size = len(word2ix)
    print(f'样本数: {len(data)}, 词典大小: {vocab_size}')

    # 划分数据集
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(torch.from_numpy(data), [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim, layer_num)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    train_losses, val_losses = train(
        train_loader, val_loader, model, optimizer, criterion, device,
        epochs=epochs, save_interval=save_interval,
        model_prefix=model_prefix, model_path=model_path
    )

    # 绘制 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss.png", dpi=150)
    plt.close()
    print("Loss curve saved to loss.png")
