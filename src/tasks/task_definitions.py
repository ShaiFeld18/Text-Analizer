from typing import Dict, Any

from src.text_analyzer.analyzer import TextAnalyzer
from src.utils.files_utils import read_json_file


class TaskDefinitions:
    """Contains all tasks"""

    def __init__(self, analyzer: TextAnalyzer):
        self.analyzer = analyzer

    def task_1(self) -> Dict[str, Any]:
        """
        Process text by the basic steps:
            1) lowercase
            2) remove punctuation
            3) remove consecutive whitespaces
            4) strip
            5) split to list of words
            6) remove unwanted words
            7) remove empty sentences and names
            8) remove duplicate names (not sentences)
        :return: processed sentences and names
        """
        return {
            "Processed Sentences": self.analyzer.sentences,
            "Processed Names": self.analyzer.persons
        }

    def task_2(self, maxk: int) -> Dict[str, Any]:
        """Count sequences up to maxk length"""
        return {
            f"{maxk}-Seq Counts": self.analyzer.count_sequences(maxk)
        }

    def task_3(self) -> Dict[str, Any]:
        """Count person mentions in text"""
        return {
            "Name Mentions": self.analyzer.count_person_mentions()
        }

    def task_4(self, qsek_query_path: str) -> Dict[str, Any]:
        """Search for sequences in text"""
        return {
            "K-Seq Matches": self.analyzer.search_sequences_from_file_in_text(
                path_to_sequences=qsek_query_path)
        }

    def task_5(self, maxk: int) -> Dict[str, Any]:
        """Analyze person contexts"""
        return {
            "Person Contexts and K-Seqs": self.analyzer.people_context(seq_len=maxk)
        }

    def task_6(self, windowsize: int, threshold: int) -> Dict[str, Any]:
        """Find connections between persons"""
        return {
            "Pair Matches": self.analyzer.find_connections(windowsize, threshold)
        }

    def task_7(self, pairs_path: str, windowsize: int, threshold: int, maximal_distance: int) -> Dict[str, Any]:
        """Find indirect connections between persons"""
        pairs_raw = [sorted(pair) for pair in read_json_file(pairs_path)["keys"]]
        found_paths = self.analyzer.indirect_connections(
            pairs_to_check=pairs_raw,
            window_size=windowsize,
            threshold=threshold,
            maximal_distance=maximal_distance
        )
        return {"Pair Matches": found_paths}

    def task_8(self, pairs_path: str, windowsize: int, threshold: int, fixed_length: int) -> Dict[str, Any]:
        """Find fixed length paths between persons"""
        pairs_raw = [sorted(pair) for pair in read_json_file(pairs_path)["keys"]]
        found_paths = self.analyzer.fixed_length_paths(
            pairs_to_check=pairs_raw,
            window_size=windowsize,
            threshold=threshold,
            maximal_distance=fixed_length,
            k=fixed_length
        )
        return {"Pair Matches": found_paths}
