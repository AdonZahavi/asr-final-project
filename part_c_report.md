# ASR Final Report

## Project goal

The project evaluates the Hebrew ASR model `ivrit-ai/whisper-large-v3-turbo-ct2` on the Mozilla Common Voice Hebrew test set.

## Part A

For Part A, I transcribed the `test.tsv` split and saved the results in:

- `results_part_a.tsv`

The output format is:

- `Filename`
- `Reference Text`
- `Transcribed Text`

All text files were handled as UTF-8.

## Part B: baseline evaluation

The baseline evaluation was computed from the raw transcription output using word-level alignment and edit statistics.

- Input file: `results_part_a.tsv`
- Output file: `results_part_b.tsv`
- Alignment log: `part_b_alignment_log.tsv`
- Frequent errors report: `part_b_frequent_errors.tsv`

Final baseline results on the clean benchmark:

- WER: `0.3592`
- Recall: `0.6578`
- Precision: `0.6554`
- F1: `0.6566`

## Part C: error analysis and normalization

### Analysis method

I analyzed the model output in two complementary ways:

1. I inspected the full alignment log for individual utterances.
2. I reviewed the most frequent error pairs over the full benchmark.

This made it possible to separate true ASR errors from artifacts caused by the evaluation procedure.

### Main issues found

The raw evaluation penalized the system for many differences that do not reflect real recognition failures:

- punctuation differences
- Hebrew vowel marks, diacritics, and quote marks
- spelling variants that keep the same meaning
- orthographic variants of common Hebrew words
- some number expressions written as words in one side and digits in the other

The goal was to change only surface form differences and same-meaning variants, as required by the assignment, and not to rewrite genuinely different content.

### Final normalization rules

The final normalization pipeline performs:

1. removal of Hebrew vowel marks and other diacritics
2. removal of punctuation and quote-like symbols
3. whitespace normalization
4. normalization of common Hebrew word variants that preserve the same meaning
5. normalization of frequent numeric and percentage expressions

### Final clean-benchmark results

- Input file: `results_part_a.tsv`
- Output file: `results_part_c.tsv`
- Alignment log: `part_c_alignment_log.tsv`
- Frequent errors report: `part_c_frequent_errors.tsv`

Final normalized results on the clean benchmark:

- WER: `0.0595`
- Recall: `0.9438`
- Precision: `0.9451`
- F1: `0.9444`

This reached the assignment target of single-digit WER.

## Part D: noisy audio robustness

For the noise condition, the assignment requires the `mod 6 = 0` setting:

- background type: noise
- SNR range: `0-6 dB`

Only long files from `MUSAN/noise/free-sound/` were used, as required. Each Common Voice test clip was downsampled to `16 kHz` using `scipy.signal.decimate(..., q=2)` before mixing.

Generated files:

- augmentation log: `augmentation_log.tsv`
- noisy transcription file: `results_part_a_noisy.tsv`
- noisy normalized evaluation file: `results_part_c_noisy.tsv`
- noisy alignment log: `part_c_noisy_alignment_log.tsv`
- noisy frequent errors report: `part_c_noisy_frequent_errors.tsv`

Final normalized results on the noisy benchmark:

- WER: `0.1705`
- Recall: `0.8414`
- Precision: `0.8452`
- F1: `0.8433`

### Interpretation

The model remains usable under strong background noise, but performance drops substantially relative to the clean benchmark:

- clean normalized WER: `0.0595`
- noisy normalized WER: `0.1705`

The increase in WER shows that background noise causes more substitutions, deletions, and insertions, especially function words and short words.

## Conclusion

The raw baseline significantly underestimated the real quality of the model because it penalized many formatting and orthographic differences. After targeted normalization based on same-meaning word variants, removal of vowel marks, and removal of punctuation, the clean-benchmark WER dropped from `0.3592` to `0.0595`. Under strong background noise, the normalized WER increased to `0.1705`, showing a clear but expected robustness degradation.
