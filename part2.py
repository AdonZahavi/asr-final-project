from __future__ import annotations

import argparse

from evaluation import AccuracyStatistics, WordEditWeights, evaluate_results_file, sequences_align


DEFAULT_INPUT_FILE = "results_part_a.tsv"
DEFAULT_OUTPUT_FILE = "results_part_b.tsv"
DEFAULT_ALIGNMENT_LOG = "part_b_alignment_log.tsv"
DEFAULT_ERRORS_REPORT = "part_b_frequent_errors.tsv"


def process_results_part_b(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
    alignment_log: str = DEFAULT_ALIGNMENT_LOG,
    errors_report: str = DEFAULT_ERRORS_REPORT,
) -> AccuracyStatistics:
    stats = evaluate_results_file(
        input_file,
        output_file,
        normalize_text=False,
        alignment_log_path=alignment_log,
        errors_report_path=errors_report,
    )
    print(f"Results written to {output_file}")
    print(f"Alignment log written to {alignment_log}")
    print(f"Frequent errors report written to {errors_report}")
    print(f"TOTAL WER: {stats.wer:.4f}")
    print(f"TOTAL Recall: {stats.recall:.4f}")
    print(f"TOTAL Precision: {stats.precision:.4f}")
    print(f"TOTAL F1: {stats.f1:.4f}")
    print("\nTop frequent errors:")
    for (ref_word, hyp_word), count in stats.frequent_errors(10):
        print(f'{count:>5}  "{ref_word}" -> "{hyp_word}"')
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part B: compute ASR accuracy metrics.")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Part A TSV file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Metrics TSV file.")
    parser.add_argument("--alignment-log", default=DEFAULT_ALIGNMENT_LOG, help="Detailed alignment TSV log.")
    parser.add_argument("--errors-report", default=DEFAULT_ERRORS_REPORT, help="Frequent errors TSV report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_results_part_b(args.input, args.output, args.alignment_log, args.errors_report)
