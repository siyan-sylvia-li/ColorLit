import os

import stanza
import glob
import json
from nltk.stem import WordNetLemmatizer
import tqdm
import argparse

nlp = stanza.Pipeline('en')
lemmatizer = WordNetLemmatizer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    for f in tqdm.tqdm(glob.glob(os.path.join(args.input_dir, "*.jsonl"))):
        js_color = open(f).readlines()
        final_lines = open(f.replace(args.input_dir, args.output_dir), "w+")
        for l in js_color:
            ld = json.loads(l)
            new_l = {}
            k = list(ld.keys())[0]
            l_sent = ld[k]['line']
            wd = ld[k]['word']
            new_l.update({"color": k})
            new_l.update({"line": l_sent})
            new_l.update({"word": wd})
            doc = nlp(l_sent)
            deps = doc.sentences[0].dependencies
            for hd, dep, sub in deps:
                sub_tx = lemmatizer.lemmatize(sub.text.lower())
                hd_tx = lemmatizer.lemmatize(hd.text.lower())
                if (sub_tx == wd and (hd.pos == "NOUN" or hd.pos == "PROPN")) \
                        or (hd_tx == wd and (sub.pos == "NOUN" or sub.pos == "PROPN")):
                    if hd_tx == wd:
                        new_l.update({"modifies": sub_tx})
                    else:
                        new_l.update({"modifies": hd_tx})
                    final_lines.write(json.dumps(new_l) + "\n")
                    break