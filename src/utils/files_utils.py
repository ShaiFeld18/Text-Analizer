import json
from typing import Dict, Any

def read_json_file(path: str) -> Dict[str, Any]:
    """Read and parse a JSON file"""
    with open(path, 'r') as file:
        return json.load(file)