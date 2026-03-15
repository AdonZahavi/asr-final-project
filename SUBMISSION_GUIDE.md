# Submission Guide

## Expected deliverables

- Part A transcription TSV: `results_part_a.tsv`
- Part B evaluation TSV: `results_part_b.tsv`
- Part C normalized evaluation TSV: `results_part_c.tsv`
- Part D augmentation log TSV: `augmentation_log.tsv`
- Part D noisy transcription TSV: `results_part_a_noisy.tsv`
- Part D noisy normalized evaluation TSV: `results_part_c_noisy.tsv`
- Part C analysis document: `part_c_report.md`

## Recommended command sequence

Run all commands from the project directory.

### Part A

```bash
python part1.py --test-tsv <path-to-test.tsv> --clips-dir <path-to-clips> --output results_part_a.tsv
```

### Part B

```bash
python part2.py --input results_part_a.tsv --output results_part_b.tsv --alignment-log part_b_alignment_log.tsv --errors-report part_b_frequent_errors.tsv
```

### Part C

```bash
python part3.py --input results_part_a.tsv --output results_part_c.tsv --alignment-log part_c_alignment_log.tsv --errors-report part_c_frequent_errors.tsv
```

### Part D

```bash
python part4.py --test-tsv <path-to-test.tsv> --clips-dir <path-to-clips> --musan-dir <path-to-musan> --output-dir noisy_clips --log-file augmentation_log.tsv
python part1.py --test-tsv <path-to-test.tsv> --clips-dir noisy_clips --audio-extension .wav --output results_part_a_noisy.tsv
python part3.py --input results_part_a_noisy.tsv --output results_part_c_noisy.tsv --alignment-log part_c_noisy_alignment_log.tsv --errors-report part_c_noisy_frequent_errors.tsv
```

## Notes for the written report

- Mention that `test.tsv` is parsed manually with `split('\t')` due to Common Voice quoting issues.
- Mention that all text files are read and written as UTF-8.
- Mention that Part D uses only `noise/free-sound/` files longer than 30 seconds, as required for the `mod 6 = 0` condition.
- Mention the final normalized WER on the clean benchmark and compare it to the noisy benchmark.
