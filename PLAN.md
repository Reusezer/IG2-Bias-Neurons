# IG2-Bias-Neurons: Original IG² Implementation

## Purpose

This repository implements the **Original IG²** algorithm from "The Devil is in the Neurons" paper for bias neuron identification. It serves as a comparison baseline for the **SignedIG²** implementation in `proanti-SignedIG2`.

## Algorithm Comparison

| Aspect | Original IG² (this repo) | SignedIG² (proanti-SignedIG2) |
| ------ | ------------------------ | ----------------------------- |
| Formula | `IG²(d1) - IG²(d2)` | Separate Pro & Anti scores |
| Output | Single gap score per neuron | Two scores per neuron |
| Classification | Positive gap → bias toward d1 | Explicit Pro/Anti labels |

## Original IG² Formula

```
IG²(w, d) = w̄ · (1/m) · Σₖ ∂P(d | α·w̄)/∂w |_{α=k/m}

IG²_gap(w) = IG²(w, d1) - IG²(w, d2)
```

Where:

- `w̄` = original neuron activation value
- `m` = number of Riemann sum steps (default: 50)
- `α` = interpolation factor (0 → 1)
- `d1, d2` = demographic attributes (e.g., "female", "male")

## Settings

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| num_steps | 50 | Riemann sum granularity |
| alpha_max | 1.0 | Interpolation range (0 → 1) |
| baseline | zeros | Standard IG baseline |
| threshold | 0.2 × max | Neuron selection threshold |

## Demographic Pairs (Same as SignedIG²)

| Category | d1 | d2 | StereoSet Domain |
| -------- | -- | -- | ---------------- |
| gender | female | male | gender |
| ethnicity | black | white | race |
| religion | christianity | islam | religion |
| occupation | doctor | waiter | profession |

## Expected Output

```
results/
├── bert-base-cased/
│   ├── gender_female_male/
│   │   ├── ig2_scores_d1.npz    # IG² for d1
│   │   ├── ig2_scores_d2.npz    # IG² for d2
│   │   ├── ig2_gap.npz          # Gap = d1 - d2
│   │   └── bias_neurons.json    # Extracted neurons
│   ├── ethnicity_black_white/
│   └── ...
└── bert-base-uncased/
    └── ...
```

## Comparison with SignedIG²

After running both implementations on the same data:

1. **Correlation check**: `IG²_gap ≈ SignedIG²_Pro - SignedIG²_Anti`
2. **Neuron overlap**: How many neurons are identified by both methods?
3. **Effectiveness**: Which method's neurons are better for bias mitigation?

## Usage

```bash
# Run IG² analysis
python scripts/run_ig2.py \
    --model bert-base-cased \
    --category gender \
    --d1 female \
    --d2 male \
    --output_dir results/bert-base-cased/gender_female_male

# Extract bias neurons
python scripts/extract_neurons.py \
    --ig2_dir results/bert-base-cased/gender_female_male \
    --threshold 0.2
```

## HPC (Wisteria)

```bash
cd /work/gb20/b20117/IG2-Bias-Neurons
pjsub jobs/job_gender.sh
```

---

*Created: 2026-01-19*
