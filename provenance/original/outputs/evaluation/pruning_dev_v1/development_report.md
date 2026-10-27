# Structured MLP pruning: development comparison

Original BF16 versus two BF16 checkpoints with MLP width reduced from 9,728 to 9,216. No recovery training. Same frozen prompts, server settings, question identities and scoring.

| Model | Math correct / 256 | Math % | Science correct / 299 | Science % | Math length finishes |
|---|---:|---:|---:|---:|---:|
| BF16 | 230 | 89.84 | 161 | 53.85 | 8 |
| RandomChannels | 220 | 85.94 | 149 | 49.83 | 0 |
| ActivationEnergy | 233 | 91.02 | 158 | 52.84 | 1 |

## Paired differences

Gains/losses are questions on which only the first/second model is correct.

| First minus second | Task | Gains | Losses | Difference (pp) | Approximate 95% interval (pp) | Exact p, unadjusted |
|---|---|---:|---:|---:|---:|---:|
| ActivationEnergy − RandomChannels | gsm8k | 23 | 10 | +5.0781 | [0.7812, 9.3750] | 0.035082 |
| ActivationEnergy − RandomChannels | arc_challenge | 23 | 14 | +3.0100 | [-1.0033, 7.0234] | 0.187742 |
| RandomChannels − BF16 | gsm8k | 11 | 21 | -3.9062 | [-8.2031, 0.3906] | 0.110184 |
| RandomChannels − BF16 | arc_challenge | 10 | 22 | -4.0134 | [-7.6923, -0.3344] | 0.050102 |
| ActivationEnergy − BF16 | gsm8k | 8 | 5 | +1.1719 | [-1.5625, 3.9062] | 0.581055 |
| ActivationEnergy − BF16 | arc_challenge | 8 | 11 | -1.0033 | [-4.0134, 1.6722] | 0.647606 |

## Interpretation and limits

- This is reused development data from Project 1, not a fresh unseen holdout.
- Intervals use 20,000 paired question resamples; math seed 20261027, science seed 20261028. They are conditional on these three checkpoints and these evaluation settings; they do not capture random-channel seed variation.
- The comparison is exploratory: one pruning width, one random-channel seed and two tasks. P-values and individual intervals are unadjusted for multiple comparisons.
- An interval containing zero does not establish equal quality or non-inferiority.
- GSM8K uses the frozen strict answer-marker rule. A model can earn credit for an early answer and continue repeating or reach the length limit. No answers were rescored or repaired.
- Math length finishes and missing-marker diagnostics are in the JSON; they may overlap. A missing marker means the frozen extraction regex found no match, not necessarily a wrong numerical answer.
- Checkpoint size reduction does not demonstrate serving speedup. No performance measurements are derived from these evaluation runs.

Source run hashes, per-question scores and flagged response endings are preserved in development_summary.json.
