import pickle

from tensorflow.keras.preprocessing.text import Tokenizer


def fit_tokenizer(train_data, num_words):
    """
    :param train_data: data for tokenizer training
    :param num_words: the length of the dictionary we will be using
    :return: trained Keras Tokenizer
    """

    # Creating and training a tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_data)

    # Saving the Tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def preprocessing(text, max_review_len, tokenizer=None):
    """
    :param text: text to be tokenized
    :param max_review_len: the length of the sequence to which we reduce all texts
    :param tokenizer: choose either your own tokenizer or one already trained at the root of the project
    :return: processed text ready for use by a neural network
    """

    if tokenizer is None:
        # Loading the Tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

    # Apply tokenizer to text
    text = tokenizer.texts_to_sequences(text)

    # Adjust all text vectors to total length
    text = pad_sequences(text, maxlen=max_review_len, padding='post')

    return text
