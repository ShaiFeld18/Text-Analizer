import json
import string

from tasks.preprocess import PROCESSED_DATA_TYPE


def _read_sequences_file(path: str) -> list[str]:
    with open(path, 'r') as file:
        sequences = json.load(file)["keys"]
    return sequences


class SearchSequence:
    def __init__(self,
                 data: PROCESSED_DATA_TYPE,
                 path_to_sequences: str):
        self.sentences = data["Question 1"]["Processed Sentences"]
        self.sequences = _read_sequences_file(path_to_sequences)
        self.counter = self._count_sequences()

    def __str__(self):
        return self.to_json()

    def _count_sequences(self) -> list[str or list[list[str]]]:
        """
        Finds in which sentence each sequence appeared.
        :return: dict mapping sequence to a list of sentences it appeared in
        """
        sequence_lens = set([len(sequence) for sequence in self.sequences])
        mapping = {}
        for sentence in self.sentences:
            for seq_len in sequence_lens:
                k_sequences = [' '.join(sentence[i:i + seq_len])
                               for i in range(0, len(sentence) - seq_len + 1)]
                for seq in k_sequences:
                    if seq not in mapping:
                        mapping[seq] = {tuple(sentence)}
                    else:
                        mapping[seq].add(tuple(sentence))

        mapping = {seq: sorted([list(s) for s in sentences], key=lambda x: ''.join(x))
                   for seq, sentences in mapping.items() if seq != ''}

        counter = []
        for seq in self.sequences:
            seq = ' '.join(seq)
            seq = seq.lower()
            seq = ''.join([char if char not in string.punctuation else ' ' for char in seq])
            seq = ' '.join(seq.split())
            if mapping.get(seq):
                counter.append([seq, mapping.get(seq)])
        counter.sort(key=lambda x: x[0])
        return counter

    def to_json(self):
        return {
            "Question 4": {
                "K-Seq Matches": self.counter
            }
        }
