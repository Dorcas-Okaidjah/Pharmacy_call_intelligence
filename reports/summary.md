# Pharmacy Benefits-Call Intelligence: results (2026-11-05)

Data: 50 real + 1000 synthetic INSURE-Dial calls. Outcome extraction scored on 50 `real` calls.

## 1. Outcome extraction

| method           |   field_accuracy |   ci95_low |   ci95_high |   calls_fully_correct |   hallucination_rate |   n_calls |   parse_rate |   median_latency_s |   mean_input_tokens |   cost_per_1k_calls |
|:-----------------|-----------------:|-----------:|------------:|----------------------:|---------------------:|----------:|-------------:|-------------------:|--------------------:|--------------------:|
| claude-haiku-5-5 |            0.819 |      0.700 |       0.907 |                 0.700 |                0.087 |        50 |        1.000 |              1.919 |            2610.660 |                     |
| rules            |            0.324 |      0.245 |       0.409 |                 0.160 |                0.128 |        50 |      nan     |            nan     |             nan     |                 nan |

Accuracy is over fields that apply to the call; `hallucination_rate` is the share of non-applicable fields where the method still produced a value. CIs resample whole calls.

| field              |   claude-haiku-5-5 |   rules |
|:-------------------|-------------------:|--------:|
| coverage_status    |              0.913 |   0.391 |
| drug1_formulary    |              0.857 |   0.000 |
| drug1_restrictions |              1.000 |   0.000 |
| drug2_formulary    |              0.929 |   0.071 |
| drug2_restrictions |              1.000 |   0.000 |
| record_found       |              0.720 |   0.480 |

## 2. Turn-level phase classifier (TF-IDF + logistic regression)

- Grouped 5-fold CV within synthetic calls: macro-F1 0.883 ± 0.008
- Trained on synthetic, tested on real calls: macro-F1 0.515
- Synthetic-to-real gap: +0.369

## 3. Operational picture (gold labels, all calls)

- calls: 1050
- record not found rate: 51.5%
- inactive coverage rate: 14.3%
- drugs checked: 620
- not on formulary rate: 21.5%
- covered drugs needing PA or restriction: 39.8%

Most frequent extraction errors: see `outcome_errors.csv` (90 rows).