import string
from itertools import chain

import pandas as pd


def process_word(text_to_process: str,
                 words_to_remove: list[str]) -> str:
    """
    Makes basic preprocessing to sentences or word that includes:
            1) lowercase all text
            2) replace punctuation with single space
            3) remove words in remove_file
            4) remove double spaces and spaces in the beginning or end of sentences
            5) converts them to a list of words
    """
    processed_text = text_to_process.lower()
    processed_text = ''.join([char if char not in string.punctuation else ' ' for char in processed_text])
    processed_text = ' '.join([word for word in processed_text.split() if word not in words_to_remove])
    return processed_text


def read_words_to_remove_file(path: str) -> list[str]:
    return list(pd.read_csv(path).iloc[:, 0]) if path else []


def flatten_list(list_of_lists: list[list[any]]) -> list[any]:
    return [item for sublist in list_of_lists for item in sublist]