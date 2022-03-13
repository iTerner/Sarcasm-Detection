import torch.nn as nn
import torch
from torch import nn, optim
import argparse
import random
import time
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import *
from dataset import SarcasmDataset
from models import SarcasmDetectionModel
import Preprocessing
from gensim import downloader

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

rep_model = downloader.load('glove-twitter-200')
bla = Preprocessing.Preprocessing(
    path='Sarcasm_Headlines_Dataset_v2.json', rep_model=rep_model, attributes=['NN', 'JJ'])


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        sen, labels, lengths = batch
        sen = sen.to(device)
        labels = labels.to(device)
        # lengths = lengths.to(device)

        predictions = model(sen, lengths).squeeze(1)
        labels = labels.float()
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:

            sen, labels, lengths = batch
            sen = sen.to(device)
            labels = labels.to(device)
            lengths = lengths.float()
            sen = sen.float()

            predictions = model(sen, lengths).squeeze(1)
            labels = labels.float()
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


DATA_SAVE_PATH = ""
PLOT_PATH = "plots/"

# get all the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pos", default="NN", type=str)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--hidden_dim", default=256, type=int)
args = parser.parse_args()

print(args)

ATTRIBUTE = args.pos
bz = args.batch_size
dropout = args.dropout
num_epochs = args.num_epochs
num_layers = args.num_layers
hidden_dim = args.hidden_dim
lr = args.lr
input_dim = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train_sen_vec = load_file(f"{DATA_SAVE_PATH}train_sen_2_vec.pkl")
train_sen_labels = load_file(f"{DATA_SAVE_PATH}train_sen_2_label.pkl")
test_sen_vec = load_file(f"{DATA_SAVE_PATH}test_sen_2_vec.pkl")
test_sen_labels = load_file(f"{DATA_SAVE_PATH}test_sen_2_label.pkl")


train_lengths = [len(val) for val in train_sen_vec.values()]
test_lengths = [len(val) for val in test_sen_vec.values()]

train_data, train_length = pad_sentences(train_sen_vec)
test_data, test_length = pad_sentences(test_sen_vec)

# save_file_pickle(train_data, f"{DATA_SAVE_PATH}train_sen_tensor.pkl")
# save_file_pickle(train_length, f"{DATA_SAVE_PATH}train_len_tensor.pkl")
# save_file_pickle(test_data, f"{DATA_SAVE_PATH}test_sen_tensor.pkl")
# save_file_pickle(test_length, f"{DATA_SAVE_PATH}data\\test_len_tensor.pkl")

train_dataset = SarcasmDataset(list(train_data.values()), list(
    train_sen_labels.values()), list(train_length.values()))
test_dataset = SarcasmDataset(list(test_data.values()), list(
    test_sen_labels.values()), list(test_length.values()))

train_dataloader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bz, shuffle=False)

model = SarcasmDetectionModel(input_dim, hidden_dim, num_layers, dropout)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)


print("start training on glove representation")
train_losses, train_accs = [], []
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss, train_acc = train(
        model, train_dataloader, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.3f}%")
print("finished training on glove representation")

# evaluate the model
print("evaluating the model on the glove representation")
test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}%")
# plot(list(range(num_epochs)), train_losses, "Loss",
#      os.path.join(PLOT_PATH))
# plot(list(range(num_epochs)), train_accs, "Accuracy",
#      os.path.join(PLOT_PATH))


print("testing projected vectors")
projected_sen_vec = load_file(
    f"{DATA_SAVE_PATH}test_sen_2_vec_{ATTRIBUTE}.pkl")
projected_sen_labels = load_file(
    f"{DATA_SAVE_PATH}test_sen_2_label_{ATTRIBUTE}.pkl")

projected_lengths = [len(val) for val in test_sen_vec.values()]

projected_data, projected_length = pad_sentences(projected_sen_vec)


projected_dataset = SarcasmDataset(list(projected_data.values()), list(
    projected_sen_labels.values()), list(projected_length.values()))

projected_dataloader = DataLoader(
    projected_dataset, batch_size=bz, shuffle=False)


print("evaluating the model on the projected representation")
test_loss, test_acc = evaluate(
    model, projected_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}%")
