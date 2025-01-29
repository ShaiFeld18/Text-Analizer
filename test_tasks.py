import pytest
import json
import os
from pathlib import Path
from main import create_parser
from src.tasks.task_runner import TaskRunner

EXAMPLES_PATH = os.path.join('tests/examples')
REMOVE_WORDS_PATH = os.path.join('tests/data', 'REMOVEWORDS.csv')

# Additional arguments for specific tasks
additional_args_by_task = {
    "2": {
        "1": ["--maxk", "3"],
        "2": ["--maxk", "4"],
        "3": ["--maxk", "5"]
    },
    "4": {
        "1": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_1", "kseq_query_keys_1.json")],
        "2": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_2", "kseq_query_keys_2.json")],
        "3": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_3", "kseq_query_keys_3.json")],
        "4": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_4", "kseq_query_keys_4.json")],
    },
    "5": {
        "1": ["--maxk", "3"],
        "2": ["--maxk", "4"],
        "3": ["--maxk", "5"],
        "4": ["--maxk", "6"]
    },
    "6": {
        "1": ["--windowsize", "4", "--threshold", "4"],
        "2": ["--windowsize", "3", "--threshold", "2"],
        "3": ["--windowsize", "5", "--threshold", "2"],
        "4": ["--windowsize", "5", "--threshold", "1"]
    },
    "7": {
        "1": ["--pairs", os.path.join(EXAMPLES_PATH, "Q7_examples", "example_1", "people_connections_1.json"),
              "--windowsize", "5", "--threshold", "2", "--maximal_distance", "1000"],
        "2": ["--pairs", os.path.join(EXAMPLES_PATH, "Q7_examples", "example_2", "people_connections_2.json"),
              "--windowsize", "3", "--threshold", "2", "--maximal_distance", "1000"],
        "3": ["--pairs", os.path.join(EXAMPLES_PATH, "Q7_examples", "example_3", "people_connections_3.json"),
              "--windowsize", "5", "--threshold", "1", "--maximal_distance", "1000"],
        "4": ["--pairs", os.path.join(EXAMPLES_PATH, "Q7_examples", "example_4", "people_connections_4.json"),
              "--windowsize", "5", "--threshold", "2", "--maximal_distance", "1000"],
    },
    "8": {
        "1": ["--pairs", os.path.join(EXAMPLES_PATH, "Q8_examples", "example_1", "people_connections_1.json"),
              "--windowsize", "5", "--threshold", "2", "--fixed_length", "2"],
        "2": ["--pairs", os.path.join(EXAMPLES_PATH, "Q8_examples", "example_2", "people_connections_2.json"),
              "--windowsize", "3", "--threshold", "2", "--fixed_length", "3"],
        "3": ["--pairs", os.path.join(EXAMPLES_PATH, "Q8_examples", "example_3", "people_connections_3.json"),
              "--windowsize", "5", "--threshold", "1", "--fixed_length", "8"]
    }
}

def parse_args(args_list):
    """Parse arguments using the parser from main"""
    parser = create_parser()
    return parser.parse_args(args_list)

def get_test_cases():
    test_cases = []
    for question_num in range(1, 9):
        question_path = Path(os.path.join(EXAMPLES_PATH, f"Q{question_num}_examples"))
        for example in question_path.iterdir():
            if example.name.startswith("example"):
                test_cases.append((question_num, example, example.name[-1]))
    return test_cases

@pytest.mark.parametrize("question_num,example_path,example_num", get_test_cases())
def test_task(question_num, example_path, example_num):
    """Test individual task implementation"""
    # Prepare arguments for the task
    args = ["-t", str(question_num), "-r", str(REMOVE_WORDS_PATH)]
    args.extend(additional_args_by_task.get(str(question_num), {}).get(example_num, []))

    # Add optional files if they exist
    people_file = example_path / f"people_small_{example_num}.csv"
    sentences_file = example_path / f"sentences_small_{example_num}.csv"

    if people_file.exists():
        args.extend(["-n", str(people_file)])
    if sentences_file.exists():
        args.extend(["-s", str(sentences_file)])

    # Parse arguments and run the task
    parsed_args = parse_args(args)
    results = TaskRunner(parsed_args).run_task()

    # Load expected solution
    result_file = list(example_path.glob(f"Q{question_num}_result*.json"))[0]
    with result_file.open('r') as file:
        expected_results = json.load(file)

    # Compare results with expected output
    assert results == expected_results, (
        f"\nExpected: {expected_results}\n"
        f"Got: {results}"
    )