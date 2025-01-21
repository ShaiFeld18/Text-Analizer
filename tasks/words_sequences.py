import os
from collections import Counter

from tasks.preprocess import PROCESSED_DATA_TYPE, Preprocessor


class SequenceFinder:
    def __init__(self,
                 data: PROCESSED_DATA_TYPE,
                 n: int):
        self.data = data
        self.n = n
        self.sequences = []
        self._count_sequences()

    def _count_sequences_by_len(self,
                                seq_len: int) -> list[list[str or int]]:
        sequences = []
        for sentence in self.data.processed_sentences:
            sequences += [' '.join(sentence[i:i + seq_len]) for i in range(0, len(sentence) - seq_len + 1)]
        sequences = [[seq, cnt] for seq, cnt in Counter(sequences).items()]
        sequences.sort(key=lambda x: x[0])
        return sequences

    def _count_sequences(self):
        self.sequences = [[f"{sequence_len}_seq", self._count_sequences_by_len(sequence_len)]
                          for sequence_len in range(1, self.n + 1)]

    def to_json(self):
        return {
            "Question 2": {
                f"{self.n}-Seq Counts": self.sequences
            }
        }
