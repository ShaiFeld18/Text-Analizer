from collections import Counter

from tasks.preprocess import PROCESSED_DATA_TYPE

Sequence_Finder_TYPE = dict[str, dict[str, list[str or int]]]


class SequenceFinder:
    def __init__(self,
                 data: PROCESSED_DATA_TYPE,
                 max_k: int):
        self.data = data
        self.n = max_k
        self.sequences = []
        self._count_sequences()

    def __str__(self):
        print(self.to_json())

    def _count_sequences_by_len(self,
                                seq_len: int) -> list[list[str or int]]:
        sequences = []
        for sentence in self.data["Question 1"]["Processed Sentences"]:
            sequences += [' '.join(sentence[i:i + seq_len]) for i in range(0, len(sentence) - seq_len + 1)]
        sequences = [[seq, cnt] for seq, cnt in Counter(sequences).items()]
        sequences.sort(key=lambda x: x[0])
        return sequences

    def _count_sequences(self):
        self.sequences = [[f"{sequence_len}_seq", self._count_sequences_by_len(sequence_len)]
                          for sequence_len in range(1, self.n + 1)]

    def to_json(self) -> Sequence_Finder_TYPE:
        return {
            "Question 2": {
                f"{self.n}-Seq Counts": self.sequences
            }
        }
