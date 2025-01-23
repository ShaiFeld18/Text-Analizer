from tasks.search_sequence import SearchSequence
from tasks.words_sequences import SequenceFinder





class ContextFinder:
    def __init__(self,
                 sentences: list[str],
                 names: list[list[[str or list[str]]]],
                 max_k: int,
                 words_to_remove: list[str]):
        self.sentences = sentences
        self.names = names
        self.max_k = max_k
        self.words_to_remove = words_to_remove

    def __str__(self):
        return self.to_json()

    def _find_sentences_with_names(self):
        names_to_sentences = {}
        for person in self.names:
            names_to_find = list(set(person[0] + [word for nickname in person[1] for word in nickname]))
            sentences_with_name = SearchSequence(self.sentences,
                                                 [n.split() for n in names_to_find],
                                                 self.words_to_remove).find_sequences()
            sentences_with_name = [sentences[1] for sentences in sentences_with_name]
            sentences_with_name = set([tuple(sentence) for sentences in sentences_with_name for sentence in sentences])
            if len(sentences_with_name) > 0:
                names_to_sentences[' '.join(person[0])] = [list(sentence) for sentence in sentences_with_name]
        return names_to_sentences

    def find_context(self):
        names_to_sentences = self._find_sentences_with_names()
        names_to_context = {}
        for name, sentences in names_to_sentences.items():
            sequences = SequenceFinder(sentences, self.max_k).find_sequences()
            sequences = [k[1] for k in sequences]
            sequences = [seq for k_seq in sequences for seq in k_seq]
            sequences = [seq[0].split() for seq in sequences]
            sequences.sort()
            names_to_context[name] = sequences
        names_to_context = dict(sorted(names_to_context.items()))
        return [[name, contexts] for name, contexts in names_to_context.items()]

    def to_json(self):
        return {
            "Question 5": {
                "Person Contexts and K-Seqs": self.find_context()
            }
        }
