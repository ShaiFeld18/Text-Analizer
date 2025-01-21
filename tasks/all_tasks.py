import json

from tasks.name_counter import NameCounter
from tasks.preprocess import Preprocessor
from tasks.words_sequences import SequenceFinder

tasks_mapping = {
    "1": lambda args: task_1(args),
    "2": lambda args: task_2(args),
    "3": lambda args: task_3(args)
}


def task_1(args):
    return Preprocessor(args.sentences, args.names, args.removewords)


def task_2(args):
    if args.preprocessed is not None:
        with open(args.preprocessed, 'r') as file:
            data = json.load(file)
    else:
        data = Preprocessor(args.sentences, args.names, args.removewords).to_json()
    return SequenceFinder(
        data=data,
        max_k=args.maxk
    )


def task_3(args):
    return NameCounter(
        data = Preprocessor(args.sentences, args.names, args.removewords).to_json()
    )