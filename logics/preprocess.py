import string
from copy import deepcopy

import pandas as pd


class Preprocessor:
    def __init__(self,
                 question_number: int,
                 sentences_path: str,
                 peoples_path: str,
                 remove_words_path: str = None
                 ):
        self.question_number = question_number
        self.sentences: pd.DataFrame = pd.read_csv(sentences_path)
        self.names: pd.DataFrame = pd.read_csv(peoples_path)
        self.remove_words: list[str] = list(pd.read_csv(remove_words_path).iloc[:, 0]) if remove_words_path else []
        self.processed_sentences: list[list[str]] = []
        self.processed_names: list[list[list[str]]] = []
        self._process_sentences()
        self._process_names()

    def _common_process(self, col: pd.Series) -> pd.Series:
        """
        Makes basic preprocessing to sentences or word that includes:
            1) lowercase all text
            2) replace punctuation with single space
            3) remove words in remove_file
            4) remove double spaces and spaces in the beginning or end of sentences
        :param col: a series representing sentences or words
        :return: series after preprocessing
        """
        processed_col = col.fillna("")
        processed_col = processed_col.apply(str.lower)  # lowercase
        processed_col = processed_col.apply(  # replace punctuation with single space
            lambda s: ''.join([char if char not in string.punctuation else ' ' for char in s])
        )
        processed_col = processed_col.apply(  # remove words in remove_file
            lambda s: ' '.join([word for word in s.split() if word not in self.remove_words])
        )
        processed_col = processed_col.apply(lambda s: ' '.join(s.split()))  # remove spaces
        return processed_col

    def _process_sentences(self):
        """
        preprocess using the basic preprocessing and parses as follows:
            1. Parsing the data as a list of sentences.
            2. Each sentence represented as a list of words.
        """
        sentences = deepcopy(self.sentences)
        sentences = sentences.apply(self._common_process)
        sentences = sentences.iloc[:, 0].apply(str.split)  # make each sentence a list
        self.processed_sentences = sentences.to_list()  # convert DataFrame to list

    def _process_names(self):
        """
        preprocess using the basic preprocessing and parses as follows:
            1. Parsing the data as a list of names.
            2. Each name is represented as a list of lists:
               - The first list contains the real name.
               - The second list contains any additional names associated with the real name.
        it also ensures that no duplicate names are present.
        """
        names = deepcopy(self.names)
        names = names.apply(self._common_process, axis=1)
        names.drop_duplicates(subset=names.columns[0], keep="first", inplace=True)
        names[names.columns[0]] = names[names.columns[0]].apply(str.split)
        names = names.values.tolist()
        #TODO: fix additional names parsing
        self.processed_names = names

    def to_json(self) -> dict[str, list[list[str]] or list[list[list[str]]]]:
        return {
            f"Question {self.question_number}": {
                "Processed Sentences": self.processed_sentences,
                "Processed Names": self.processed_names
            }
        }
