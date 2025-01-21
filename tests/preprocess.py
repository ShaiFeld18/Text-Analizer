import json
import os
import sys

import pytest

from logics.preprocess import Preprocessor


def test_preprocessor():

    with open('examples/Q1_examples/example_1/Q1_result1.json', 'r') as file:
        sol = json.load(file)
    res = Preprocessor(question_number=1,
                       sentences_path="examples/Q1_examples/example_1/sentences_small_1.csv",
                       peoples_path="examples/Q1_examples/example_1/people_small_1.csv",
                       remove_words_path="data/REMOVEWORDS.csv"
                       )
    assert res.to_json() == sol


if __name__ == '__main__':
    pytest.main()
