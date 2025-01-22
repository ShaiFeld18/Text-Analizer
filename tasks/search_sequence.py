import json

from tasks.preprocess import PROCESSED_DATA_TYPE
from tasks.utils import process_word, read_words_to_remove_file


def _read_sequences_file(path: str) -> list[str]:
    with open(path, 'r') as file:
        sequences = json.load(file)["keys"]
    return sequences


class SearchSequence:
    def __init__(self,
                 data: PROCESSED_DATA_TYPE,
                 path_to_sequences: str,
                 path_to_remove_words: str):
        self.sentences = data["Question 1"]["Processed Sentences"]
        self.sequences = _read_sequences_file(path_to_sequences)
        self.remove_words = read_words_to_remove_file(path_to_remove_words)
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
        sequences_to_check = [process_word(' '.join(seq), self.remove_words) for seq in self.sequences]
        sequences_in_counter = []
        for seq in sequences_to_check:
            if seq not in sequences_in_counter and mapping.get(seq):
                counter.append([seq, mapping.get(seq)])
                sequences_in_counter.append(seq)
        counter.sort(key=lambda x: x[0])
        return counter

    def to_json(self):
        return {
            "Question 4": {
                "K-Seq Matches": self.counter
            }
        }
