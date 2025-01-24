import json

from analyzer import TextAnalyzer


def _start_analyzer(args) -> TextAnalyzer:
    if args.preprocessed is not None:
        with open(args.preprocessed, 'r') as file:
            data = json.load(file)
            analyzer = TextAnalyzer(sentences=data["Question 1"]["Processed Sentences"],
                                    persons=data["Question 1"]["Processed Names"],
                                    path_to_unwanted_words=args.removewords)
    else:
        analyzer = TextAnalyzer(path_to_sentences=args.sentences,
                                path_to_persons=args.names,
                                path_to_unwanted_words=args.removewords)
    return analyzer


class TaskRunner:
    def __init__(self, args):
        self.analyzer = _start_analyzer(args)
        self.args = args

    def task_1(self):
        return {f"Question 1": {"Processed Sentences": self.analyzer.sentences,
                                "Processed Names": self.analyzer.persons}
                }

    def task_2(self):
        return self.analyzer.count_sequences(self.args.maxk)

    def task_3(self):
        return self.analyzer.count_person_mentions()

    def task_4(self):
        return self.analyzer.search_sequences_from_file_in_text(
            path_to_sequences=self.args.qsek_query_path
        )

    def task_5(self):
        return self.analyzer.people_context(seq_len=self.args.maxk)
