from typing import List, Set


class GraphAnalyzer:
    """Handles graph operations for finding paths"""

    @staticmethod
    def find_all_paths(graph: List[List[str]],
                       start: str,
                       end: str,
                       max_len: int) -> List[List[str]]:
        """
        Find all paths between two nodes in an undirected graph with maximum length.

        :param graph: List of edges where each edge is a list of two node names.
        :param start: Starting node name.
        :param end: Destination node name.
        :param max_len: Maximum allowed length of a path.
        :return: List of all possible paths from start to end
        """

        def get_neighbors(node: str) -> List[str]:
            """Find all neighbors of a given node in the graph."""
            neighbors = []
            for edge in graph:
                if edge[0] == node:
                    neighbors.append(edge[1])
                elif edge[1] == node:
                    neighbors.append(edge[0])
            return neighbors

        def dfs(current: str, path: List[str], paths: List[List[str]], visited: Set[str]) -> None:
            """recursive function for dfs."""
            if len(path) > max_len:
                return

            if current == end:
                paths.append(path[:])
                return

            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # find all paths from this node
                    dfs(neighbor, path + [neighbor], paths, visited)
                    # because we want all paths and not just one
                    visited.remove(neighbor)

        paths: List[List[str]] = []
        visited: Set[str] = {start}
        dfs(start, [start], paths, visited)

        return paths
