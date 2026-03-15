from __future__ import annotations

import argparse

from common_voice import DEFAULT_TEST_TSV, read_test_rows


def explore_tsv(test_tsv: str = str(DEFAULT_TEST_TSV), preview_rows: int = 5) -> None:
    rows = read_test_rows(test_tsv)
    if not rows:
        print(f"No rows found in {test_tsv}")
        return

    header = list(rows[0].keys())
    print(f"Exploring: {test_tsv}")
    print(f"Total samples: {len(rows)}")
    print("-" * 40)
    print("Columns:")
    for index, column in enumerate(header):
        print(f"[{index}] {column}")
    print("-" * 40)

    for idx, row in enumerate(rows[:preview_rows], start=1):
        print(f"Sample #{idx}")
        for key in header:
            print(f"  {key}: {row.get(key, '')}")
        print("-" * 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Common Voice Hebrew test.tsv.")
    parser.add_argument("--test-tsv", default=str(DEFAULT_TEST_TSV), help="Path to Common Voice test.tsv.")
    parser.add_argument("--preview-rows", type=int, default=5, help="Number of rows to preview.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explore_tsv(args.test_tsv, args.preview_rows)
