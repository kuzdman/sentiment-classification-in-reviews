#!/usr/bin/env python

import pandas as pd
import numpy as np

from keras.models import load_model

from preprocessing import fit_tokenizer, preprocessing
from training import train_model


def main():
    text = [
        "The coffee was not as good as I wanted. I don't understand why it's so bitter.",
        "The food is amazing! Beautiful place! I recommend!"
    ]
    text = pd.Series(np.array(text))
    text = preprocessing(text)

    model = load_model('model')
    predictions = model.predict(text)

    for i in range(len(predictions)):
        print(f"Review {i + 1} -> {round(predictions[i, 0] * 100, 1)}% good review")


if __name__ == "__main__":
    main()
