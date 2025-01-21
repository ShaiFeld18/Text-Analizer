from tasks.preprocess import Preprocessor
from tasks.words_sequences import SequenceFinder

tasks_mapping = {
    "1": lambda args: task_1(args),
    "2": lambda args: task_2(args),
}


def task_1(args):
    return Preprocessor(args.sentences, args.names, args.removewords).to_json()


def task_2(args):
    return SequenceFinder(
        data=Preprocessor(args.sentences, args.names, args.removewords).to_json(),
        max_k=args.maxk
    ).to_json()
