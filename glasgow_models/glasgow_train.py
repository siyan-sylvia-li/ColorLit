import os
import pickle

import pandas
import torch
from torch.utils.data import Dataset
import fasttext
import fasttext.util
from glasgow_model import GlasgowFC
import numpy as np
from scipy.stats import pearsonr
import argparse


bce = torch.nn.BCELoss()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
fasttext.util.download_model('en', if_exists='ignore')


class GlasgowDataset(Dataset):
    def __init__(self, im_df, lab, wv_p=None):
        self.words = list(im_df["Words"])
        self.wv = []
        self.labels = torch.FloatTensor(list(im_df[lab]))
        # need to normalize
        max_lab = torch.max(self.labels)
        min_lab = torch.min(self.labels)
        self.labels = (self.labels - min_lab) / (max_lab - min_lab)
        self.ft = fasttext.load_model('cc.en.300.bin')
        fasttext.util.reduce_model(self.ft, FT_DIM)
        if not wv_p:
            self.add_fast_text()
        else:
            self.wv = wv_p

    def add_fast_text(self):
        for w in self.words:
            self.wv.append(self.ft.get_word_vector(w))
        assert len(self.wv) == len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.FloatTensor(self.wv[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)


def train(model, train_loader):
    global optimizer
    model.train()
    run_loss = 0
    pearson_corr = 0
    for id, (data, labels) in enumerate(train_loader):
        # print(data.shape)
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)

        total_loss = bce(output, labels.unsqueeze(-1).to(device))
        pearson_corr += pearsonr(output.squeeze().detach().numpy(), labels.detach().numpy()).statistic
        total_loss.backward(retain_graph=True)

        run_loss += total_loss.detach().item()
        optimizer.step()

    run_total_loss = run_loss / len(train_loader)
    pearson_corr = pearson_corr / len(train_loader)

    print("TRAIN Total Loss = {l4}; Pearson Correlation = {r1} ".format(
        l4=run_total_loss, r1=pearson_corr))
    return run_total_loss


def eval(model, val_loader):
    model.eval()
    run_total_loss = 0
    pearson_corr = 0

    for id, (data, labels) in enumerate(val_loader):
        data = data.to(device)
        out = model(data)
        total_loss = bce(out, labels.unsqueeze(-1).to(device))
        run_total_loss += total_loss.detach().item()
        pearson_corr += pearsonr(out.squeeze().detach().numpy(), labels.detach().numpy()).statistic
    run_total_loss = run_total_loss / len(val_loader)
    pearson_corr = pearson_corr / len(val_loader)

    print("EVAL Total Loss = {l4}; Pearson Correlation = {r1} ".format(
        l4=run_total_loss, r1=pearson_corr))
    return run_total_loss


def print_test(model, test_loader):
    model.eval()

    for id, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        out = model(data)
        for l, o in zip(labels, out):
            print(l, o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glasgow_label", type=str)
    parser.add_argument("--train", action="store_true", default=False)
    args = parser.parse_args()

    FT_DIM = 100
    BATCH_SIZE = 128
    LR = 0.005
    N_HIDDEN = 128
    GLASGOW_LABEL = args.glasgow_label
    im_df = pandas.read_csv("glasgow.csv")

    # create directory
    if not os.path.exists("models/"):
        os.mkdir("models/")

    if not args.train:
        model = torch.load("models/best_loss_{}_model.pt".format(GLASGOW_LABEL))
    else:
        model = GlasgowFC(n_feat=FT_DIM, n_hidden=N_HIDDEN)

    model.float()
    model.to(device)

    tr_set = pandas.read_csv("train_set.csv")
    val_set = pandas.read_csv("val_set.csv")
    test_set = pandas.read_csv("test_set.csv")
    print(len(tr_set), len(val_set), len(test_set))

    if not os.path.exists("train_wv.p"):
        img_train = GlasgowDataset(tr_set, GLASGOW_LABEL)
        pickle.dump(img_train.wv, open("train_wv.p", "wb+"))
    else:
        train_wv = pickle.load(open("train_wv.p", "rb"))
        img_train = GlasgowDataset(tr_set, GLASGOW_LABEL, train_wv)
    train_loader = torch.utils.data.DataLoader(img_train, batch_size=BATCH_SIZE)

    if not os.path.exists("val_wv.p"):
        img_val = GlasgowDataset(val_set, GLASGOW_LABEL)
        pickle.dump(img_val.wv, open("val_wv.p", "wb+"))
    else:
        val_wv = pickle.load(open("val_wv.p", "rb"))
        img_val = GlasgowDataset(val_set, GLASGOW_LABEL, val_wv)
    val_loader = torch.utils.data.DataLoader(img_val, batch_size=BATCH_SIZE)

    if not os.path.exists("test_wv.p"):
        img_test = GlasgowDataset(test_set, GLASGOW_LABEL)
        pickle.dump(img_test.wv, open("test_wv.p", "wb+"))
    else:
        test_vw = pickle.load(open("test_wv.p", "rb"))
        img_test = GlasgowDataset(test_set, GLASGOW_LABEL, test_vw)
    test_loader = torch.utils.data.DataLoader(img_test, batch_size=BATCH_SIZE)

    if args.train:
        best_loss = 1e3
        best_loss_model = None
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
        for epoch in range(800):
            train(model, train_loader)
            ev_loss = eval(model, val_loader)
            if ev_loss < best_loss:
                best_loss = ev_loss
                best_loss_model = model
        print("========= START TESTING TRAINED MODEL =============")
        print_test(model, test_loader)
        eval(model, test_loader)
        torch.save(model, "models/trained_{}_model.pt".format(GLASGOW_LABEL))
        torch.save(best_loss_model, "models/best_loss_{}_model.pt".format(GLASGOW_LABEL))
    else:
        print("========= START TESTING BEST MODEL =============")
        print_test(model, test_loader)
        eval(model, test_loader)

