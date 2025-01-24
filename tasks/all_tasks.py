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

    def run_task(self):
        task_map = {
            "1": self._task_1,
            "2": self._task_2,
            "3": self._task_3,
            "4": self._task_4,
            "5": self._task_5,
            "6": self._task_6,
            "7": self._task_7
        }
        task_num = self.args.task
        if task_num not in task_map:
            raise ValueError(f"Invalid task number: {task_num}")

        return {f"Question {task_num}": task_map[task_num]()}

    def _task_1(self):
        return {
            "Processed Sentences": self.analyzer.sentences,
            "Processed Names": self.analyzer.persons
        }

    def _task_2(self):
        return {
            f"{self.args.maxk}-Seq Counts": self.analyzer.count_sequences(self.args.maxk)
        }

    def _task_3(self):
        return {
            "Name Mentions": self.analyzer.count_person_mentions()
        }

    def _task_4(self):
        return {
            "K-Seq Matches": self.analyzer.search_sequences_from_file_in_text(
                path_to_sequences=self.args.qsek_query_path)
        }

    def _task_5(self):
        return {
            "Person Contexts and K-Seqs": self.analyzer.people_context(seq_len=self.args.maxk)
        }

    def _task_6(self):
        return {
            "Pair Matches": self.analyzer.find_connections(self.args.windowsize,
                                                           self.args.threshold)
        }

    def _task_7(self):
        return self.analyzer.indirect_connections(
            pairs_to_check=self.args.pairs,
            window_size=self.args.windowsize,
            threshold=self.args.threshold,
            maximal_distance=self.args.maximal_distance
        )
