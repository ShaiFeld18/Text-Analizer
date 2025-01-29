import argparse

from src.tasks.task_runner import TaskRunner


def create_parser():
    parser = argparse.ArgumentParser(
        prog='Text Analyzer project',
    )
    # General arguments
    parser.add_argument('-t', '--task',
                        help="task number",
                        required=True)
    parser.add_argument('-s', '--sentences',
                        help="Sentence file path")
    parser.add_argument('-n', '--names',
                        help="Names file path")
    parser.add_argument('-r', '--removewords',
                        help="Words to remove file path")
    parser.add_argument('-p', '--preprocessed',
                        action='append',
                        help="json with preprocessed data",
                        default=None)

    # Task specific arguments
    parser.add_argument('--maxk',
                        type=int,
                        help="Max k")
    parser.add_argument('--fixed_length',
                        type=int,
                        help="fixed length to find")
    parser.add_argument('--windowsize',
                        type=int,
                        help="Window size")
    parser.add_argument('--pairs',
                        help="json file with list of pairs")
    parser.add_argument('--threshold',
                        type=int,
                        help="graph connection threshold")
    parser.add_argument('--maximal_distance',
                        type=int,
                        help="maximal distance between nodes in graph")
    parser.add_argument('--qsek_query_path',
                        help="json file with query path")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    result = TaskRunner(args).run_task()
    print(result)


if __name__ == '__main__':
    main()
