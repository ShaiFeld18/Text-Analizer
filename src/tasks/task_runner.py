from typing import Dict, Any

from .task_definitions import TaskDefinitions
from ..text_analyzer.analyzer import TextAnalyzer
from ..utils.files_utils import read_json_file


class TaskRunner:
    """Task runner"""

    def __init__(self, args):
        self.analyzer = self._initialize_analyzer(args)
        self.task_definitions = TaskDefinitions(self.analyzer)
        self.args = args

    @staticmethod
    def _initialize_analyzer(args) -> TextAnalyzer:
        """Initialize the analyzer based on command line arguments"""
        if args.preprocessed is not None:
            data = read_json_file(args.preprocessed)
            return TextAnalyzer(
                sentences=data["Question 1"]["Processed Sentences"],
                persons=data["Question 1"]["Processed Names"],
                path_to_unwanted_words=args.removewords
            )
        return TextAnalyzer(
            path_to_sentences=args.sentences,
            path_to_persons=args.names,
            path_to_unwanted_words=args.removewords
        )

    def run_task(self) -> Dict[str, Any]:
        """Run the specified task"""
        task_map = {
            "1": self.task_definitions.task_1,
            "2": lambda: self.task_definitions.task_2(self.args.maxk),
            "3": self.task_definitions.task_3,
            "4": lambda: self.task_definitions.task_4(self.args.qsek_query_path),
            "5": lambda: self.task_definitions.task_5(self.args.maxk),
            "6": lambda: self.task_definitions.task_6(self.args.windowsize, self.args.threshold),
            "7": lambda: self.task_definitions.task_7(
                self.args.pairs,
                self.args.windowsize,
                self.args.threshold,
                self.args.maximal_distance
            ),
            "8": lambda: self.task_definitions.task_8(
                self.args.pairs,
                self.args.windowsize,
                self.args.threshold,
                self.args.fixed_length
            )
        }

        task_num = self.args.task
        if task_num not in task_map:
            raise ValueError(f"Invalid task number: {task_num}")

        return {f"Question {task_num}": task_map[task_num]()}
