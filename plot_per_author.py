import glob

import numpy
import os
import torch
import json
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import umap
import gensim.downloader
from collections import Counter


if __name__ == "__main__":
    """
        You can change the author names
    """
    A1 = "Joyce_James"
    A2 = "Fitzgerald_F._Scott_"

    # create directory
    if not os.path.exists("author_plots/"):
        os.mkdir("author_plots/")

    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
    hiddens = {}
    meta_info = {}
    all_words = {}
    to_iter = glob.glob("litbank_color_lines/*{}*.jsonl".format(A1)) + glob.glob("litbank_color_lines/*{}*.jsonl".format(A2))
    for f in to_iter:
        jf = open(f).readlines()
        print(f)
        seen_word = set()
        for l in jf:
            ld = json.loads(l)
            color = ld['color']
            aname = f.split("/")[-1].replace(".jsonl", "")
            aname = "_".join(aname.split("_")[2:])
            if ld['modifies'] not in seen_word:
                try:
                    outputs = glove_vectors[ld['modifies']]
                    if color not in hiddens:
                        hiddens.update({color: []})
                        meta_info.update({color: []})
                    hiddens[color].append(outputs.reshape(-1, 1))
                    meta_info[color].append((f.split("/")[-1].replace(".jsonl", ""), color, ld['modifies']))
                    seen_word.add(ld['modifies'])
                    if aname not in all_words:
                        all_words.update({aname: []})
                    all_words[aname].append(ld['modifies'])
                except KeyError:
                    continue
    for an in all_words:
        all_words[an] = Counter(all_words[an])
    reducer = umap.UMAP()
    for c in hiddens:
        for i in range(len(hiddens[c])):
            hiddens[c][i] = hiddens[c][i].reshape(1, -1)
        all_embeds = numpy.concatenate(hiddens[c], axis=0)
        all_embeds = reducer.fit_transform(all_embeds)
        hiddens[c] = all_embeds

    pickle.dump(hiddens, open("all_hiddens.p", "wb+"))
    pickle.dump(meta_info, open("meta_info.p", "wb+"))
    pickle.dump(all_words, open("all_words.p", "wb+"))

    hiddens = pickle.load(open("all_hiddens.p", "rb"))
    meta_info = pickle.load(open("meta_info.p", "rb"))
    all_words = pickle.load(open("all_words.p", "rb"))

    figure(figsize=(10, 8), dpi=400)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    # reducer = umap.UMAP()
    for c in hiddens:
        all_embeds = hiddens[c]
        meta = meta_info[c]
        pts = []
        color_authors = []
        markers = []
        sizes = []
        annots = []
        for j, m in enumerate(meta):
            if A1 in m[0]:
                color_authors.append("blue")
                markers.append("x")
                sizes.append(all_words[A1][m[-1]] * 15)
                pts.append(all_embeds[j, :])
                annots.append(m[-1])
                print(A1, m, all_embeds[j], all_words[A1][m[-1]])
            elif A2 in m[0]:
                color_authors.append("red")
                markers.append("o")
                print(A2, m, all_embeds[j], all_words[A2][m[-1]])
                sizes.append(all_words[A2][m[-1]] * 15)
                pts.append(all_embeds[j, :])
                annots.append(m[-1])
        pts = numpy.array(pts)
        plt.scatter(x=pts[:, 0], y=pts[:, 1], c=color_authors, s=sizes, alpha=0.4)
        for i in range(len(pts)):
            plt.annotate(text=annots[i], xy=(pts[i, 0], pts[i, 1]))
        plt.title(c)
        plt.savefig("author_plots/" + c + "_cluster_ft_{a1}_{a2}.png".format(a1=A1, a2=A2), bbox_inches='tight', dpi=400)
        plt.clf()


