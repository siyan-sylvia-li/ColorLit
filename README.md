# ColorLit

This repo contains the code and data for the paper Color Me Intrigued: Quantifying Usage of Colors in Fiction, accepted into the Creative AI Across Modalities workshop, AAAI 2023.

## **Top level `.py` files** 
   * `classify_glasgow.py`: Given an input file and a Glasgow Norm Label (in this work we focus on `IMAG`, `CNC`, and `VAL`), load the corresponding trained Glasgow norm model and predict the corresponding value for each word.
   * `compute_glasgow_means.py`: Compute the average Glasgow Norm values per author.
   * `plots.py`: Plot the separate Glasgow Norm plots over time.
   * `plot_per_author.py` (not shown in paper): Plot the GloVe embeddings of color-dependent nouns between two different authors.
## **`final_data` folder** 
All the data used for plotting; the final average Glasgow Norm values are in the `*_author_means.csv` files for each type of Glasgow Norm labels.
## **`litbank_color_lines` folder** 
Contains the lines containing color words (or synonyms) and the dependent noun extracted from all LitBank novels.

## **`color_extraction` folder**
`find_dependent_noun.py`: Use stanza to extract color-dependent nouns; assuming the input file contains all individual sentences containing a color word / synonym of interest.

## `glasgow_models` folder
Training and evaluation code for the Glasgow Norm MLP's. The trained models are also stored in this directory.
