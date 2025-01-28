import string
import pandas as pd
import re
from typing import List, Optional


class TextProcessor:
    """Handles all text processing operations"""

    @staticmethod
    def process_string(string_to_process: str) -> str:
        """Makes basic preprocessing to sentences or word"""
        processed_text = string_to_process.lower()
        processed_text = ''.join([char if char not in string.punctuation else ' '
                                  for char in processed_text])
        processed_text = re.sub(r'\s+', ' ', processed_text)
        processed_text = processed_text.strip()
        return processed_text

    @staticmethod
    def process_unwanted_words(path_to_unwanted_words: Optional[str]) -> List[str]:
        """Process and return list of unwanted words"""
        return list(pd.read_csv(path_to_unwanted_words).iloc[:, 0]) if path_to_unwanted_words else []

    @staticmethod
    def get_sequences_from_sentence(sentence: List[str], seq_len: int) -> List[str]:
        """Extract sequences of given length from a sentence"""
        return [' '.join(sentence[i:i + seq_len]) for i in range(len(sentence) - seq_len + 1)]
