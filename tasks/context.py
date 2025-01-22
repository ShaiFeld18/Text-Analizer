from tasks.search_sequence import SearchSequence
from tasks.words_sequences import SequenceFinder


def _find_name_context(sentence: list[str], name: str, k: int) -> list[list[str]]:
    indices = [i for i, word in enumerate(sentence) if word == name]
    contexts = []
    for index in indices:
        start = max(0, index - k)
        end = min(len(sentence), index + k + 1)
        context = ' '.join(sentence[start:end])
        contexts.append(context)

    return [context.split() for context in contexts]


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
            names_to_find = person[0] + [word for nickname in person[1] for word in nickname]
            names_to_find = names_to_find + [' '.join(person[0])] + [' '.join(nickname) for nickname in person[1]]
            names_to_find = list(set(names_to_find))
            sentences_with_name = SearchSequence(self.sentences,
                                                 [name.split() for name in names_to_find],
                                                 self.words_to_remove).find_sequences()
            for res in sentences_with_name:
                names_to_sentences[res[0]] = res[1]
        return names_to_sentences

    def find_context(self):
        names_to_sentences = self._find_sentences_with_names()
        names_to_context = {}
        for name, sentences in names_to_sentences.items():
            contexts = [_find_name_context(sentence, name, self.max_k) for sentence in sentences]
            contexts = [item for sublist in contexts for item in sublist]
            sequences = SequenceFinder(contexts, self.max_k).find_sequences()
            all_sequences = []
            for k in sequences:
                for seq in k[1]:
                    all_sequences.append(seq[0])
            names_to_context[name] = sorted(all_sequences)
        names_to_context = dict(sorted(names_to_context.items()))
        return [[name, contexts] for name, contexts in names_to_context.items()]

    def to_json(self):
        return {
            "Question 5": {
                "Person Contexts and K-Seqs": self.find_context()
            }
        }
