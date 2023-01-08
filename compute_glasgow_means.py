import json
import pandas
import numpy as np

if __name__ == "__main__":
    data = {}
    """
        Change this to IMAG or CNC or VAL
    """
    GLASGOW_LABEL = "VAL"
    aimg_dict = json.load(open("author_{}_dict.json".format(GLASGOW_LABEL)))

    for a in aimg_dict:
        color_line = []
        for c in ["red", "green", "black", "white", "blue", "brown", "gray", "yellow", "pink", "purple"]:
            imageabilities = np.array(list(set(aimg_dict[a][c])))
            print(len(imageabilities))
            if len(imageabilities) >= 1:
                img_mean = np.mean(imageabilities)
                img_std = np.std(imageabilities)
            else:
                img_mean = 0
                img_std = 0
            color_line.append(img_mean)
            color_line.append(img_std)
        data.update({a: color_line})

    cols = []
    for c in ["red", "green", "black", "white", "blue", "brown", "gray", "yellow", "pink", "purple"]:
        cols.append(c + "_mean")
        cols.append(c + "_std")
    df = pandas.DataFrame.from_dict(data, orient="index", columns=cols)
    df.to_csv("{}_author_means.csv".format(GLASGOW_LABEL))