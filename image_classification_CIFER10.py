import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import tensorboard

# --- データの準備 ---

# データを読み込んだ後の行う処理の定義
transform = transforms.Compose([
  transforms.ToTensor(),  # torch.Tensor型に変換する必要があるため
])

# CIFER10
train_val = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# シードを固定する
torch.manual_seed(0)

# trainとvalを 0.8 : 0.2 に分割する
n_train = int(len(train_val) * 0.8)
n_val = len(train_val) - n_train
train, val = torch.utils.data.random_split(train_val, [n_train, n_val])
len(train), len(val)

# DataLoaderを用意する
batch_size = 256

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


# --- ネットワークの定義 ---

class Net(pl.LightningModule):
  def __init__(self):
    super().__init__()

    # 畳み込み層を定義
    self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), padding=(1, 1))
    # 全結合層を定義
    self.fc = nn.Linear(1536, 10)

    self.train_acc = pl.metrics.Accuracy()
    self.val_acc = pl.metrics.Accuracy()
    self.test_acc = pl.metrics.Accuracy()

  def forward(self, x):
    # 畳み込み
    h = self.conv(x)
    # 最大値プーリング
    h = F.max_pool2d(h, kernel_size=(2, 2), stride=2)
    # ReLU関数
    h = F.relu(h)
    # ベクトル化
    h = h.view(-1, 1536)
    # 線型結合
    h = self.fc(h)
    return h

  def training_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('train_loss', loss, on_step=True, on_epoch=True)
    # self.log('train_acc', self.train_acc(y, t), on_step=True, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('val_loss', loss, on_step=False, on_epoch=True)
    # self.log('val_acc', self.val_acc(y, t), on_step=False, on_epoch=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('test_loss', loss, on_step=False, on_epoch=True)
    # self.log('test_acc', self.test_acc(y, t), on_step=False, on_epoch=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    return optimizer


# --- 学習 ---
pl.seed_everything(0)
net = Net()
trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(net, train_loader, val_loader)


# --- テスト ---
results = trainer.test(test_dataloaders=test_loader)
print(results)


# TensorBoard
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
