import copy
import json

import matplotlib.pyplot as plt
import pandas
import scipy.stats
from matplotlib.pyplot import figure

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glasgow_label", type=str)
    args = parser.parse_args()
    GLASGOW_LABEL = args.glasgow_label

    im_df = pandas.read_csv("final_data/{}_author_means.csv".format(GLASGOW_LABEL), index_col=[0])
    # years = list(im_df['Year'])
    years = list(im_df.index)
    years = [int((x.split("_"))[1]) for x in years]
    pearson_corr = {}
    figure(figsize=(5.5, 4.5), dpi=400)
    plt.rcParams.update({'font.size': 20})
    plt.tight_layout()
    # genders = list(im_df['Gender'])
    for c in ["red", "green", "black", "white", "blue", "brown", "gray", "yellow", "pink", "purple"]:
        means = list(im_df[c + "_mean"])
        stds = list(im_df[c + "_std"])
        x_p, y_p = [], []
        for y, m in zip(years, means):
            if m > 0:
                x_p.append(y)
                y_p.append(m)
        res = scipy.stats.pearsonr(x_p, y_p)
        pearson_corr.update({c: {"pr": res.statistic, "pval": res.pvalue}})
        print(c, res.statistic, res.pvalue)

        if c == "white":
            plot_c = "cyan"
        elif c == "yellow":
            plot_c = "gold"
        else:
            plot_c = c

        plt.scatter(x=x_p, y=y_p, marker="o", color=plot_c)
        plt.title(c)
        plt.xlabel("Year")
        plt.ylabel(GLASGOW_LABEL)
        plt.savefig("{gl}_{c}.png".format(gl=GLASGOW_LABEL, c=c), bbox_inches='tight', dpi=400)
        plt.clf()
    json.dump(pearson_corr, open("{}_pearson.json".format(GLASGOW_LABEL), "w+"))