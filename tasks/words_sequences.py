from collections import Counter

Sequence_Finder_TYPE = dict[str, dict[str, list[str or int]]]


class SequenceFinder:
    def __init__(self,
                 sentences: list[list[str]],
                 max_k: int):
        self.sentences = sentences
        self.n = max_k

    def __str__(self):
        print(self.to_json())

    def _count_sequences_by_len(self,
                                seq_len: int) -> list[list[str or int]]:
        sequences = []
        for sentence in self.sentences:
            sequences += [' '.join(sentence[i:i + seq_len]) for i in range(0, len(sentence) - seq_len + 1)]
        sequences = [[seq, cnt] for seq, cnt in Counter(sequences).items()]
        sequences.sort(key=lambda x: x[0])
        return sequences

    def find_sequences(self) -> list[list[str or int]]:
        return [[f"{sequence_len}_seq", self._count_sequences_by_len(sequence_len)]
                for sequence_len in range(1, self.n + 1)]

    def to_json(self) -> Sequence_Finder_TYPE:
        return {
            "Question 2": {
                f"{self.n}-Seq Counts": self.find_sequences()
            }
        }
