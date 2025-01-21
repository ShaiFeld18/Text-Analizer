import string
from copy import deepcopy

import pandas as pd

PROCESSED_DATA_TYPE = dict[str, dict[str, list[list[str]] or list[list[list[str]]]]]


class Preprocessor:
    def __init__(self,
                 sentences_path: str,
                 peoples_path: str or None,
                 remove_words_path: str = None
                 ):
        self.sentences: pd.DataFrame = pd.read_csv(sentences_path)
        self.names: pd.DataFrame = pd.read_csv(peoples_path) if peoples_path else None
        self.remove_words: list[str] = list(pd.read_csv(remove_words_path).iloc[:, 0]) if remove_words_path else []
        self.processed_sentences = self._process_sentences()
        self.processed_names = self._process_names() if self.names is not None else None

    def __str__(self):
        print(self.to_json())

    def _process_text(self, sentence: str) -> list[str] or None:
        """
        Makes basic preprocessing to sentences or word that includes:
            1) lowercase all text
            2) replace punctuation with single space
            3) remove words in remove_file
            4) remove double spaces and spaces in the beginning or end of sentences
            5) converts them to a list of words
        :param sentence: string with a sentence or name
        :return: list of words after preprocessing
        """
        sentence = sentence.lower()
        sentence = ''.join([char if char not in string.punctuation else ' ' for char in sentence])
        sentence = ' '.join([word for word in sentence.split() if word not in self.remove_words])
        return sentence.split()

    def _process_sentences(self) -> list[list[str]]:
        """
        preprocess each words using the basic preprocessing and parses as follows:
            1. Parsing the data as a list of sentences.
            2. Each sentence represented as a list of words.
        """
        sentences = deepcopy(self.sentences)
        sentences = sentences[sentences.columns[0]]
        sentences.fillna('', inplace=True)
        sentences = sentences.apply(self._process_text)
        sentences = sentences[sentences.apply(lambda x: x != [])]
        return sentences.to_list()

    def _process_names(self) -> list[list[list[str]]]:
        """
        preprocess using the basic preprocessing and parses as follows:
            1. Parsing the data as a list of names.
            2. Each name is represented as a list of lists:
               - The first list contains the real name.
               - The second list contains any additional names associated with the real name.
        it also ensures that no duplicate names are present.
        """
        names = deepcopy(self.names)
        names.fillna('', inplace=True)
        name_col, additional_col = names.columns[0], names.columns[1]
        names[name_col] = names[name_col].apply(self._process_text)
        names['full_name'] = names[name_col].apply(lambda x: ' '.join(x))
        names.drop_duplicates(subset='full_name', keep="first", inplace=True)
        names.drop('full_name', axis=1, inplace=True)
        names[additional_col] = names[additional_col].apply(
            lambda x: [self._process_text(name) for name in x.split(',')]
        )
        names[additional_col] = names[additional_col].apply(lambda x: x if len(x[0]) > 0 else [])
        names = names.values.tolist()
        return names

    def to_json(self) -> PROCESSED_DATA_TYPE:
        return {
            "Question 1": {
                "Processed Sentences": self.processed_sentences,
                "Processed Names": self.processed_names
            }
        }
