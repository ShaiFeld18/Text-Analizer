from typing import List


class GraphAnalyzer:
    """Handles graph operations"""

    @staticmethod
    def find_all_paths(graph: List[List[List[str]]],
                       start: List[str],
                       end: str,
                       max_len: int) -> List[List[List[str]]]:
        """
        Find all paths between two nodes in a graph with maximum length.
        :param graph: graph of connected people.
        :param start: starting node.
        :param end: destination node.
        :param max_len: max length of a path.
        :return: List of all possible paths from start to end up to max length.
        """
        paths = []
        stack = [(start, [start])]  # Stack holds tuples of (current_node, current_path)

        while stack:
            current, path = stack.pop()

            if len(path) > max_len:
                continue
            if current == end:
                paths.append(path)
                continue

            for edge in graph:
                if (edge[0] == current and edge[1] not in path) or (edge[1] == current and edge[0] not in path):
                    next_node = edge[1] if edge[0] == current else edge[0]
                    stack.append((next_node, path + [next_node]))

        return paths
