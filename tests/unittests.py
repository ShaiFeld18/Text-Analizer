import json
import os

from main import read_args
from tasks.all_tasks import tasks_mapping

EXAMPLES_PATH = os.path.join('..', 'examples')
REMOVE_WORDS_PATH = os.path.join('..', 'data', 'REMOVEWORDS.csv')

# Additional arguments for specific tasks and examples
additional_args_by_task = {
    "2": {
        "1": ["--maxk", "3"],
        "2": ["--maxk", "4"],
        "3": ["--maxk", "5"]
    }
}

def test_all_tasks():
    """Run tests for all tasks and verify results against expected outputs."""
    for question_num in range(1, 4):
        question_path = os.path.join(EXAMPLES_PATH, f"Q{question_num}_examples")

        for example in os.listdir(question_path):
            example_path = os.path.join(question_path, example)
            example_num = example[-1]

            # Prepare arguments for the task
            args = ["-t", str(question_num), "-r", REMOVE_WORDS_PATH]
            args.extend(additional_args_by_task.get(str(question_num), {}).get(str(example_num), []))

            # Add optional files if they exist
            if f"people_small_{example_num}.csv" in os.listdir(example_path):
                args.extend(["-n", os.path.join(example_path, f"people_small_{example_num}.csv")])
            if f"sentences_small_{example_num}.csv" in os.listdir(example_path):
                args.extend(["-s", os.path.join(example_path, f"sentences_small_{example_num}.csv")])

            # Parse arguments and run the task
            args = read_args(args)
            results = tasks_mapping[args.task](args).to_json()

            # Load expected solution and compare
            solution_path = os.path.join(question_path, example, f"Q{question_num}_result{example_num}.json")
            with open(solution_path, 'r') as file:
                expected_results = json.load(file)

            assert results == expected_results, (
                f"Test failed for question {question_num}, example {example}. "
                f"Expected {expected_results}, got {results}."
            )

            print(f"Tested question {question_num} with example {example} successfully.")
