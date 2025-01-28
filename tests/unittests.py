import json
import os
from pathlib import Path

from main import create_parser
from src.tasks.task_runner import TaskRunner

EXAMPLES_PATH = os.path.join('examples')
REMOVE_WORDS_PATH = os.path.join('data', 'REMOVEWORDS.csv')

# Additional arguments for specific tasks and examples
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


def run_single_test(question_num: int, example_path: Path, example_num: str) -> dict:
    """
    Run a single test case and return the results.

    Args:
        question_num: The question number being tested
        example_path: Path to the example directory
        example_num: The example number being tested

    Returns:
        dict containing test results and any error information
    """
    try:
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
        result_files = list(example_path.glob(f"Q{question_num}_result*.json"))
        if not result_files:
            raise FileNotFoundError(f"No result file found for Q{question_num} in {example_path}")

        with result_files[0].open('r') as file:
            expected_results = json.load(file)

        return {
            "success": results == expected_results,
            "expected": expected_results,
            "result": results,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "expected": None,
            "result": None,
            "error": str(e)
        }


def test_all_tasks():
    """Run tests for all tasks and verify results against expected outputs."""
    all_results = []

    for question_num in range(1, 9):
        question_path = Path(os.path.join(EXAMPLES_PATH, f"Q{question_num}_examples"))

        if not question_path.exists():
            print(f"Warning: No examples found for question {question_num}")
            continue

        for example in question_path.iterdir():
            if example.is_dir() and example.name.startswith("example"):
                example_num = example.name[-1]

                test_result = run_single_test(question_num, example, example_num)
                test_result["question"] = question_num
                test_result["example"] = example.name
                all_results.append(test_result)

                if test_result["success"]:
                    print(f"✓ Question {question_num} {example.name} passed")
                else:
                    print(f"✗ Question {question_num} {example.name} failed")
                    if test_result["error"]:
                        print(f"  Error: {test_result['error']}")
                    else:
                        print(f"  Expected: {test_result['expected']}")
                        print(f"  Got:      {test_result['result']}")

    # Print summary
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r["success"])

    print(f"\nTest Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    return all_results


if __name__ == "__main__":
    test_all_tasks()
