import torch
import pandas
import argparse
import fasttext.util
import tqdm
import json


def build_img_dict(g_df):
    """
    Normalize values of the desired Glasgow Norm Label to be in [0, 1]
    :param g_df: the dataframe of glasgow.csv
    """
    max_la = max(list(g_df[GLASGOW_LABEL]))
    min_la = min(list(g_df[GLASGOW_LABEL]))
    for i, row in g_df.iterrows():
        glasgow_dict.update({row["Words"]: (row[GLASGOW_LABEL] - min_la) / (max_la - min_la)})


def find_imag(word):
    """
    Use the glasgow norm model to classify the glasgow label value of the word queried.
    :param word: the word whose {IMAG, VAL, CNC} value we want to know.
    :return: the {IMAG, VAL, CNC} value.
    """
    if word in glasgow_dict:
        in_dict.append(word)
        return glasgow_dict[word]
    wv = torch.FloatTensor(ft.get_word_vector(word)).unsqueeze(0)
    out = img_model(wv).squeeze().item()
    out_dict.append(word)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glasgow_label", type=str)
    parser.add_argument("--input_file", type=str, default="author_color_dict.json")
    args = parser.parse_args()
    FT_DIM = 100
    GLASGOW_LABEL = args.glasgow_label
    in_dict = []
    out_dict = []
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, FT_DIM)
    glasgow_dict = {}
    glasgow_df = pandas.read_csv("glasgow.csv")
    build_img_dict(glasgow_df)
    img_model = torch.load("glasgow_models/models/best_loss_{}_model.pt".format(GLASGOW_LABEL))
    acd = json.load(open('author_color_dict.json'))
    aimg_dict = {}
    for a in tqdm.tqdm(acd):
        aimg_dict.update({a: {}})
        for c in ["red", "green", "black", "white", "blue", "brown", "gray", "yellow", "pink", "purple"]:
            if c in acd[a]:
                imag_list = []
                for w in acd[a][c]["word"]:
                    imag_list.append(find_imag(w[1]))
                aimg_dict[a].update({c: imag_list})
            else:
                aimg_dict[a].update({c: []})
    json.dump(aimg_dict, open("author_{}_dict.json".format(GLASGOW_LABEL), "w+"))
    print(len(set(in_dict)))
    print(len(set(out_dict)))