from typing import List, Set, Tuple, Dict


class GraphAnalyzer:
    """Handles graph-related operations for text analysis"""

    @staticmethod
    def find_all_paths(graph: List[List[str]], start: str, end: str, max_len: int) -> List[List[str]]:
        """Find all paths between two nodes in a graph with maximum length constraint"""

        def dfs(current: str, path: List[str]) -> List[List[str]]:
            if len(path) > max_len:
                return []
            if current == end:
                return [path]

            paths = []
            for edge in graph:
                if (edge[0] == current and edge[1] not in path) or \
                        (edge[1] == current and edge[0] not in path):
                    next_node = edge[1] if edge[0] == current else edge[0]
                    new_paths = dfs(next_node, path + [next_node])
                    paths.extend(new_paths)
            return paths

        return dfs(start, [start])
