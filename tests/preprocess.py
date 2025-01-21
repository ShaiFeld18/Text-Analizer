import json
import os

import pytest

from logics.preprocess import Preprocessor

EXAMPLES_PATH = os.path.join('..', 'examples')
REMOVE_WORDS_PATH = os.path.join('..', 'data', 'REMOVEWORDS.csv')


def test_question1():
    question_path = os.path.join(EXAMPLES_PATH, f"Q1_examples")
    for example in os.listdir(question_path):
        example_num = example[-1]
        results = Preprocessor(question_number=1,
                               sentences_path=os.path.join(question_path, example,
                                                           f"sentences_small_{example_num}.csv"),
                               peoples_path=os.path.join(question_path, example, f"people_small_{example_num}.csv"),
                               remove_words_path=REMOVE_WORDS_PATH
                               )
        with open(os.path.join(question_path, example, f"Q1_result{example_num}.json"), 'r') as file:
            sol = json.load(file)
        assert results.to_json() == sol


if __name__ == '__main__':
    pytest.main([__file__])
