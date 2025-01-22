import json
import os
import urllib

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
    },
    "4": {
        "1": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_1", "kseq_query_keys_1.json")],
        "2": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_2", "kseq_query_keys_2.json")],
        "3": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_3", "kseq_query_keys_3.json")],
        "4": ["--qsek_query_path", os.path.join(EXAMPLES_PATH, "Q4_examples", "example_4", "kseq_query_keys_4.json")],
    }
}



def test_all_tasks():
    """Run tests for all tasks and verify results against expected outputs."""
    failed_tests = []  # Store details of failed tests

    for question_num in [1, 2, 4]:
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

            if results != expected_results:
                failed_tests.append({
                    "question": question_num,
                    "example": example,
                    "expected": expected_results,
                    "result": results
                })
            else:
                print(f"Tested question {question_num} with example {example} successfully.")

    # Print summary of failed tests
    if failed_tests:
        print("\nSome tests failed:")
        for test in failed_tests:
            print(
                f"Question {test['question']} Example {test['example']}: \nExpected: {test['expected']}\nActual:   {test['result']}")
    else:
        print("\nAll tests passed successfully!")


# Run the tests
if __name__ == "__main__":
    test_all_tasks()
