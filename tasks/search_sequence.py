from tasks.utils import process_word


class SearchSequence:
    def __init__(self,
                 sentences: list[str],
                 sequences: list[list[str]],
                 remove_words: list[str]):
        self.sentences = sentences
        self.sequences = sequences
        self.remove_words = remove_words

    def __str__(self):
        return self.to_json()

    def find_sequences(self) -> list[str or list[list[str]]]:
        """
        Finds in which sentence each sequence appeared.
        :return: list mapping sequence to a list of sentences it appeared in
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
                "K-Seq Matches": self.find_sequences()
            }
        }
