from typing import List, Any

def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list"""
    return [item for sublist in list_of_lists for item in sublist]