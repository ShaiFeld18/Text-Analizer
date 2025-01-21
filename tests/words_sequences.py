import json
import os

import pytest

from tasks.preprocess import Preprocessor
from tasks.words_sequences import SequenceFinder

EXAMPLES_PATH = os.path.join('..', 'examples')
REMOVE_WORDS_PATH = os.path.join('..', 'data', 'REMOVEWORDS.csv')


def test_question1():
    question_path = os.path.join(EXAMPLES_PATH, f"Q2_examples")
    example_to_k = {1: 3, 2: 4, 3: 5}
    for example in os.listdir(question_path):
        example_num = example[-1]
        results = SequenceFinder(Preprocessor(sentences_path=os.path.join(question_path, example,
                                                                          f"sentences_small_{example_num}.csv"),
                                              peoples_path=os.path.join(question_path, example,
                                                                        f"people_small_{example_num}.csv"),
                                              remove_words_path=REMOVE_WORDS_PATH
                                              ),
                                 example_to_k[int(example_num)])
        with open(os.path.join(question_path, example, f"Q2_result{example_num}.json"), 'r') as file:
            sol = json.load(file)
        assert results.to_json() == sol


if __name__ == '__main__':
    pytest.main([__file__])
