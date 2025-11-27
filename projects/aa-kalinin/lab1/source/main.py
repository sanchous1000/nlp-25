import argparse
from pathlib import Path

from utils import download_nltk_data, process_dataset, demonstrate_lemmatization_ambiguity


def main():
    download_nltk_data()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='../../dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='../../assets/annotated-corpus',
                        help='Path to output directory')
    parser.add_argument('--report', action='store_true',
                        help='Generate lemmatization evaluation report')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress information')

    args = parser.parse_args()

    if args.report:
        print(demonstrate_lemmatization_ambiguity())
    else:
        script_dir = Path(__file__).parent
        dataset_dir = (script_dir / args.dataset_dir).resolve()
        output_dir = (script_dir / args.output_dir).resolve()

        print(f"Dataset directory: {dataset_dir}")
        print(f"Output directory: {output_dir}")

        process_dataset(str(dataset_dir), str(output_dir), args.verbose)
        print("\nCompleted successfully!")


if __name__ == '__main__':
    main()
