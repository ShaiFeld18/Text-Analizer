import json
import re
import string
from collections import defaultdict, deque

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


def _get_sequences_from_sentence(sentence: list[str], seq_len: int) -> list[str]:
    return [' '.join(sentence[i:i + seq_len]) for i in range(len(sentence) - seq_len + 1)]


def find_all_paths(graph, start, end, max_len):
    def dfs(current, path):
        # If path length exceeds max_len, stop exploring
        if len(path) > max_len:
            return []

        # If reached the end, return the path
        if current == end:
            return [path]

        paths = []
        # Explore neighbors
        for edge in graph:
            # Check both directions of the edge
            if (edge[0] == current and edge[1] not in path) or \
                    (edge[1] == current and edge[0] not in path):
                # Determine the next node
                next_node = edge[1] if edge[0] == current else edge[0]
                new_paths = dfs(next_node, path + [next_node])
                paths.extend(new_paths)

        return paths

    # Start the search from the start node
    return dfs(start, [start])


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

    def _map_seq_by_len_to_sentences(self, seq_len: int) -> dict[str, dict[str, list[list[str]] or int]]:
        """
        Find all sequences of given length in text.
        Find all the sentences each sequence appears and count how many timnes the sequences appeared.
        :param seq_len: length of sequences to find
        :return: a list mapping a sequence to all sentences it appears in
        """
        results = {}
        for sentence in self.sentences:
            seqs_in_sentence = _get_sequences_from_sentence(sentence, seq_len)
            for seq in seqs_in_sentence:
                results[seq] = results.get(seq, {"sentences": set(), "count": 0})
                results[seq]["sentences"].add(tuple(sentence))
                results[seq]["count"] += 1
        for seq in results.keys():
            results[seq]["sentences"] = sorted([list(s) for s in results[seq]["sentences"]])
        return results

    def count_sequences(self, seq_len: int) -> list[str or list[str or int]]:
        """
        Find all the sequences up to the given sequence length.
        Count how many times each sequence appeared in the text.
        :param seq_len: maximum length of the sequences to find and count
        :return: list mapping a sequence to the times it appeared in the text
        """
        results = []
        for sequence_len in range(1, seq_len + 1):
            seqs_to_sentences = self._map_seq_by_len_to_sentences(sequence_len)
            results.append(
                [f"{sequence_len}_seq",
                 sorted([[seq, sentences["count"]] for seq, sentences in seqs_to_sentences.items()])]
            )
        return results

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
        results = [[' '.join(sequence), mapping[' '.join(sequence)]["sentences"]] for sequence in words
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
        drop_duplicates = set([tuple(seq) for seq in sequences if len(seq) > 0])
        sequences = [list(seq) for seq in drop_duplicates]
        return self._search_sequences_in_text(sequences)

    def _map_names_to_sentences(self) -> dict[str, list[list]]:
        names_to_sentences = {}
        for person in self.persons:
            names_to_find = [[name] for name in person[0]] + [[" ".join(person[0])]] + person[1]
            sentences_with_name = self._search_sequences_in_text(names_to_find)
            sentences_with_name = [sentences[1] for sentences in sentences_with_name]
            sentences_with_name = set([tuple(sentence) for sentences in sentences_with_name for sentence in sentences])
            if len(sentences_with_name) > 0:
                names_to_sentences[' '.join(person[0])] = [list(sentence) for sentence in sentences_with_name]
        return names_to_sentences

    def people_context(self, seq_len: int) -> list[list[str or list[str]]]:
        """
        For each person find all the sentences they appear in.
        For each sentence, search all the k-len sequences in the sentence.
        :param seq_len: maximum length of the sequences to find
        :return: a list mapping a person to all the sequences in sentences they appear.
        """
        names_to_sentences = self._map_names_to_sentences()
        names_to_context = dict()
        for person, sentences in names_to_sentences.items():
            seqs = []
            for n in range(1, seq_len + 1):
                seqs.append(flatten_list([_get_sequences_from_sentence(sentence, n)
                                          for sentence in sentences]))
            seqs = [seq.split() for seq in list(set(flatten_list(seqs)))]
            seqs.sort()
            names_to_context[person] = seqs
        names_to_context = [[k, v] for k, v in names_to_context.items()]
        names_to_context.sort(key=lambda x: x[0])
        return names_to_context

    def _find_distance_between_sentences(self, a: list[str], b: list[str]) -> int:
        return abs(self.sentences.index(a) - self.sentences.index(b))

    def find_connections(self, window_size: int, threshold: int) -> list[list[list[str]]]:
        """
        Find pairs of people who appear within distinct windows of sentences.
        :param window_size: Size of windows.
        :param threshold: Minimum number of times two people must appear within a window.
        :return: List of pairs of connected persons.
        """
        persons_to_sentences = self._map_names_to_sentences()
        connections = []
        processed_pairs = set()

        windows = [
            self.sentences[i:i + window_size]
            for i in range(len(self.sentences) - window_size + 1)
        ]

        for person_a, sentences_a in persons_to_sentences.items():
            for person_b, sentences_b in persons_to_sentences.items():
                if person_a >= person_b:  # Ensure each pair is processed only once
                    continue

                pair_key = (person_a, person_b)
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                counter = 0
                for window in windows:
                    if any(sentence in window for sentence in sentences_a) and \
                            any(sentence in window for sentence in sentences_b):
                        counter += 1
                if counter >= threshold:
                    connections.append(sorted([person_a.split(), person_b.split()]))
        return sorted(connections)

    def names_to_person(self, names: list[str]) -> str:
        """
        Search names in self.persons and return the main name as string.
        :param names: list of names to search
        :return: main names of persons
        """
        actual_persons = []
        for name in names:
            for person in self.persons:
                if name in [person[0]] + [nickname for nickname in person[1]]:
                    actual_persons.append(' '.join(person[0]))
        return ' '.join(actual_persons)

    def _all_indirect_connections(self,
                                  pairs_to_check: list[list[str]],
                                  window_size: int,
                                  threshold: int,
                                  maximal_distance: int) -> dict[tuple[str, str], list[str]]:
        graph = self.find_connections(window_size, threshold)
        for pair in graph:
            pair[0], pair[1] = ' '.join(pair[0]), ' '.join(pair[1])
        results = {}
        for start, end in pairs_to_check:
            if start == '' or end == '':
                results[(start, end)] = []
                continue
            results[(start, end)] = find_all_paths(
                graph,
                start,
                end,
                maximal_distance)
        return results

    def indirect_connections(self,
                             pairs_to_check: list[list[str]],
                             window_size: int,
                             threshold: int,
                             maximal_distance: int) -> list[list[str or bool]]:
        res = self._all_indirect_connections(pairs_to_check, window_size, threshold, maximal_distance)
        for pair in pairs_to_check:
            if len(res.get(tuple(pair), [])) > 0:
                pair.append(True)
            else:
                pair.append(False)
        res = sorted(pairs_to_check, key=lambda x: x[0])
        return res

    def fixed_length_paths(self,
                           pairs_to_check: list[list[str]],
                           window_size: int,
                           threshold: int,
                           maximal_distance: int,
                           k: int) -> list[list[str or bool]]:
        res = self._all_indirect_connections(pairs_to_check, window_size, threshold, maximal_distance)
        for pair in pairs_to_check:
            for path in res[tuple(pair)]:
                if len(pair) > 2:
                    continue
                if len(path) >= k:
                    pair.append(True)
            if len(pair) == 2:
                pair.append(False)
        return pairs_to_check
