import json
import re
import string

import pandas as pd

from tasks.utils import flatten_list


def _process_string(string_to_process: str) -> str:
    """
    Makes basic preprocessing to sentences or word that includes:
            1) lowercase all text
            2) replace punctuation with single space
            3) remove double spaces
            4) remove spaces in the beginning or end of sentences
    """
    processed_text = string_to_process.lower()  # lowercase
    processed_text = ''.join([char if char not in string.punctuation else ' '
                              for char in processed_text])  # remove punctuation
    processed_text = re.sub(r'\s+', ' ', processed_text)  # remove double spaces
    processed_text = processed_text.strip()  # strip
    return processed_text


def _process_unwanted_words(path_to_unwanted_words: str or None) -> list[str]:
    return list(pd.read_csv(path_to_unwanted_words).iloc[:, 0]) if path_to_unwanted_words else []


def _get_sequences_from_sentence(sentence: list[str], seq_len: int) -> dict[str, list[list[str]]]:
    results = {}
    for i in range(len(sentence) - seq_len + 1):
        results.setdefault(' '.join(sentence[i:i + seq_len]), []).append(sentence)
    return results


class TextAnalyzer:
    def __init__(self,
                 path_to_sentences: str = None,
                 path_to_persons: str = None,
                 path_to_unwanted_words: str = None,
                 sentences: list[list[str]] = None,
                 persons: list[list[list[str]]] = None,
                 unwanted_words: list[str] = None):
        # Check if paths and processed data are mixed
        paths_provided = path_to_sentences is not None or path_to_persons is not None
        processed_data_provided = sentences is not None or persons is not None
        if paths_provided and processed_data_provided:
            raise ValueError("You must provide either paths or processed data, not both.")

        if path_to_unwanted_words is None:
            raise ValueError("You must provide path to unwanted words.")

        self.unwanted_words = _process_unwanted_words(path_to_unwanted_words)

        if processed_data_provided:
            self.sentences = sentences
            self.persons = persons

        if paths_provided:
            self.sentences = self._process_sentences(path_to_sentences)
            self.persons = self._process_persons(path_to_persons)

    def _process_sentences(self, path_to_sentences: str or None) -> list[list[str]]:
        """
        preprocess each word using the basic preprocessing and parses as follows:
            1. Parsing the data as a list of sentences.
            2. Each sentence represented as a list of words.
        it also removes empty sentences.
        """
        if path_to_sentences is None:
            return []
        sentences = pd.read_csv(path_to_sentences)
        sentences = sentences[sentences.columns[0]]
        sentences.fillna('', inplace=True)
        sentences = sentences.apply(_process_string)  # basic preprocess
        sentences = sentences.apply(str.split)  # split to list of words
        sentences = sentences.apply(lambda x: self._remove_unwanted_words(x))  # remove unwanted words
        sentences = sentences[sentences.apply(lambda x: x != [])]  # remove empty sentences
        return sentences.to_list()

    def _process_persons(self, path_to_names: str or None) -> list[list[list[str]]]:
        """
        preprocess using the basic preprocessing and parses as follows:
            1. Parsing the data as a list of names.
            2. Each name is represented as a list of lists:
               - The first list contains the real name.
               - The second list contains any additional names associated with the real name.
        it also ensures that no duplicate names are present.
        """
        if path_to_names is None:
            return []
        names = pd.read_csv(path_to_names)
        names.fillna('', inplace=True)
        name_col, additional_names_col = names.columns[0], names.columns[1]
        names[name_col] = names[name_col].apply(_process_string)  # basic preprocess
        names[name_col] = names[name_col].apply(str.split)  # split to list of words
        names[name_col] = names[name_col].apply(lambda x: self._remove_unwanted_words(x))  # remove unwanted words

        # drop duplicates
        names['full_name'] = names[name_col].apply(lambda x: ' '.join(x))
        names.drop_duplicates(subset='full_name', keep="first", inplace=True)
        names.drop('full_name', axis=1, inplace=True)

        names[additional_names_col] = names[additional_names_col].apply(
            lambda x: [self._remove_unwanted_words(  # remove unwanted words
                _process_string(name).split()  # basic preprocess + split to list of words
            ) for name in x.split(',')]
        )
        names[additional_names_col] = names[additional_names_col].apply(  # remove empty nicknames
            lambda x: x if len(x[0]) > 0 else []
        )
        names = names.values.tolist()
        return names

    def _remove_unwanted_words(self, text: list[str]) -> list[str]:
        """
        Removes unwanted words from the given text.
        :param text: sentence a list of string
        :return: text with unwanted words removed
        """
        return [word for word in text if word not in self.unwanted_words]

    def _map_seq_by_len_to_sentences(self, seq_len: int) -> dict[str, list[list[str]]]:
        """
        Find all sequences of given length in text.
        Find all the sentences each sequence appears.
        :param seq_len: length of sequences to find
        :return: a list mapping a sequence to all sentences it appears in
        """
        results = {}
        for sentence in self.sentences:
            results.update(_get_sequences_from_sentence(sentence, seq_len))
        for seq in results.keys():
            results[seq] = sorted(results[seq])
        return results

    def count_sequences(self, seq_len: int) -> list[list[str or int]]:
        """
        Find all the sequences up to the given sequence length.
        Count how many times each sequence appeared in the text.
        :param seq_len: maximum length of the sequences to find and count
        :return: list mapping a sequence to the times it appeared in the text
        """
        results = {}
        for sequence_len in range(1, seq_len + 1):
            seqs_to_sentences = self._map_seq_by_len_to_sentences(sequence_len)
            results.update({seq: len(sentences) for seq, sentences in seqs_to_sentences.items()})
        return [[k, v] for k, v in results.items()]

    def count_person_mentions(self) -> list[list[str or int]]:
        """
        Counts how many times each person appeared in the text.
        :return: list mapping person by their full name to the times it appeared in the text
        """
        text = ' '.join([' '.join(s) for s in self.sentences])
        name_counter = []
        for person in self.persons:
            full_name = ' '.join(person[0])
            names_to_count = person[0] + [word for nickname in person[1] for word in nickname]
            counter = 0
            for name in set(names_to_count):
                counter += text.count(name)
            if counter > 0:
                name_counter.append([full_name, counter])
        name_counter.sort(key=lambda x: x[0])
        return name_counter

    def _search_sequences_in_text(self, words: list[list[str]]) -> list[list[str or list[str]]]:
        """
        For each word search for all the sentences it appears in.
        We start by mapping all the words with relevant lengths to all the sentences they appear in.
        Then we just search each sequence in the mapping so the search is made in O(1) time.
        :param words: words to search
        :return: a list mapping a sequence to all the sentences it appears in the text
        """
        # Find unique sequence lengths
        sequence_lens = {len(sequence) for sequence in words}

        # Map sequences to sentences
        mapping = {}
        for seq_len in sequence_lens:
            mapping.update(self._map_seq_by_len_to_sentences(seq_len))

        # make the search
        results = [[' '.join(sequence), mapping[' '.join(sequence)]] for sequence in words
                   if mapping.get(' '.join(sequence))]
        results.sort(key=lambda x: x[0])
        return results

    def search_sequences_from_file_in_text(self, path_to_sequences: str) -> list[list[str or list[str]]]:
        """
        For each sequence, search for all the sentences it appears in.
        :param path_to_sequences: a path to a JSON file with words to find
        :return: a list mapping a sequence to all the sentences it appears in the text
        """
        # process sequences file
        with open(path_to_sequences, 'r') as file:
            sequences: list[list[str]] = json.load(file)["keys"]
        sequences = [self._remove_unwanted_words(_process_string(' '.join(seq)).split())
                     for seq in sequences]

        return self._search_sequences_in_text(sequences)

    def people_context(self, seq_len: int) -> list[list[str or list[str]]]:
        """
        For each person find all the sentences they appear in.
        For each sentence, search all the k-len sequences in the sentence.
        :param seq_len: maximum length of the sequences to find
        :return: a list mapping a person to all the sequences in sentences they appear.
        """
        names_to_sentences = {}
        for person in self.persons:
            names_to_find = list(set(person[0] + [word for nickname in person[1] for word in nickname]))
            sentences_with_name = self._search_sequences_in_text([n.split() for n in names_to_find])
            sentences_with_name = [sentences[1] for sentences in sentences_with_name]
            sentences_with_name = set([tuple(sentence) for sentences in sentences_with_name for sentence in sentences])
            if len(sentences_with_name) > 0:
                names_to_sentences[' '.join(person[0])] = [list(sentence) for sentence in sentences_with_name]

        names_to_context = dict()
        for person, sentences in names_to_sentences.items():
            seqs = []
            for n in range(1, seq_len + 1):
                seqs.append(flatten_list([list(_get_sequences_from_sentence(sentence, n).keys())
                                          for sentence in sentences]))
            seqs = flatten_list(seqs)
            seqs.sort()
            names_to_context[person] = seqs
        names_to_context = [[k, v] for k, v in names_to_context.items()]
        names_to_context.sort(key=lambda x: x[0])
        return names_to_context


if __name__ == '__main__':
    t = TextAnalyzer(
        path_to_sentences="examples/Q5_examples/example_1/sentences_small_1.csv",
        path_to_persons="examples/Q5_examples/example_1/people_small_1.csv",
        path_to_unwanted_words="data/REMOVEWORDS.csv"
    )
    # print(t.count_sequences(3))
    print(t.people_context(3))
