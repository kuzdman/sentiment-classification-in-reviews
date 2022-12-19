#!/usr/bin/env python

import pandas as pd
import numpy as np

from preprocessing import fit_tokenizer, preprocessing


def main():
    x_train = pd.read_csv('data/train.csv')

    # fit_tokenizer(x_train.Review, 50000)
    # text = preprocessing(pd.Series(np.array(['The coffee was not as good as I wanted'])), 50)

    # print(text)


if __name__ == "__main__":
    main()
