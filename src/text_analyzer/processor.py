import string
import pandas as pd
import re
from typing import List, Optional


class TextProcessor:
    """Handles all basic text processing operations"""

    @staticmethod
    def process_string(string_to_process: str) -> str:
        """
        Process text by the basic steps:
            1) lowercase
            2) remove punctuation
            3) remove consecutive whitespaces
            4) strip
        :param string_to_process: text to be processed.
        :return: processed text.
        """
        processed_text = string_to_process.lower()
        processed_text = ''.join([char if char not in string.punctuation else ' '
                                  for char in processed_text])  # remove punctuation
        processed_text = re.sub(r'\s+', ' ', processed_text)  # remove consecutive whitespaces
        processed_text = processed_text.strip()
        return processed_text

    @staticmethod
    def process_unwanted_words(path_to_unwanted_words: Optional[str]) -> List[str]:
        """
        Read a file of unwanted words.
        :param path_to_unwanted_words: path to unwanted words file.
        :return: list of unwanted words.
        """
        return list(pd.read_csv(path_to_unwanted_words).iloc[:, 0]) if path_to_unwanted_words else []

    @staticmethod
    def get_sequences_from_sentence(sentence: List[str],
                                    seq_len: int) -> List[str]:
        """
        Finds all sequences of a specific length from a sentence.
        :param sentence: sentence to be processed.
        :param seq_len: sequence length.
        :return: list of all sequences.
        """
        return [' '.join(sentence[i:i + seq_len]) for i in range(len(sentence) - seq_len + 1)]
