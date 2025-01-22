import json

from tasks.context import ContextFinder
from tasks.name_counter import NameCounter
from tasks.preprocess import Preprocessor
from tasks.search_sequence import SearchSequence
from tasks.utils import read_words_to_remove_file
from tasks.words_sequences import SequenceFinder


def _read_sequences_file(path: str) -> list[str]:
    with open(path, 'r') as file:
        sequences = json.load(file)["keys"]
    return sequences


tasks_mapping = {
    "1": lambda args: task_1(args),
    "2": lambda args: task_2(args),
    "3": lambda args: task_3(args),
    "4": lambda args: task_4(args),
    "5": lambda args: task_5(args)
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
        sentences=data["Question 1"]["Processed Sentences"],
        max_k=args.maxk
    )


def task_3(args):
    return NameCounter(
        data=Preprocessor(args.sentences, args.names, args.removewords).to_json()
    )


def task_4(args):
    return SearchSequence(
        sentences=Preprocessor(args.sentences, args.names, args.removewords).to_json()["Question 1"]["Processed Sentences"],
        sequences=_read_sequences_file(args.qsek_query_path),
        remove_words=read_words_to_remove_file(args.removewords)
    )


def task_5(args):
    if args.preprocessed is not None:
        with open(args.preprocessed, 'r') as file:
            data = json.load(file)
    else:
        data = Preprocessor(args.sentences, args.names, args.removewords).to_json()
    return ContextFinder(data["Question 1"]["Processed Sentences"],
                         data["Question 1"]["Processed Names"],
                         args.maxk,
                         read_words_to_remove_file(args.removewords))
