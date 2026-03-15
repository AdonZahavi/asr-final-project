from __future__ import annotations

import argparse

from evaluation import AccuracyStatistics, normalize, evaluate_results_file


DEFAULT_INPUT_FILE = "results_part_a.tsv"
DEFAULT_OUTPUT_FILE = "results_part_c.tsv"
DEFAULT_ALIGNMENT_LOG = "part_c_alignment_log.tsv"
DEFAULT_ERRORS_REPORT = "part_c_frequent_errors.tsv"


def process_results_part_c(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
    alignment_log: str = DEFAULT_ALIGNMENT_LOG,
    errors_report: str = DEFAULT_ERRORS_REPORT,
) -> AccuracyStatistics:
    stats = evaluate_results_file(
        input_file,
        output_file,
        normalize_text=True,
        alignment_log_path=alignment_log,
        errors_report_path=errors_report,
    )
    print(f"Results written to {output_file}")
    print(f"Alignment log written to {alignment_log}")
    print(f"Frequent errors report written to {errors_report}")
    print(f"TOTAL WER after normalization: {stats.wer:.4f}")
    print("\nTop frequent errors after normalization:")
    for (ref_word, hyp_word), count in stats.frequent_errors(20):
        print(f'{count:>5}  "{ref_word}" -> "{hyp_word}"')
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part C: normalize text and recompute metrics.")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Part A TSV file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Normalized metrics TSV file.")
    parser.add_argument("--alignment-log", default=DEFAULT_ALIGNMENT_LOG, help="Detailed alignment TSV log.")
    parser.add_argument("--errors-report", default=DEFAULT_ERRORS_REPORT, help="Frequent errors TSV report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_results_part_c(args.input, args.output, args.alignment_log, args.errors_report)
