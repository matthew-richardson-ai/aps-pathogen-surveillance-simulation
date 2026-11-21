# Justin's mixed APS dataset

Run from the project root:

```powershell
python Justin/generate_dataset.py --samples 10000
```

Each run writes one new timestamped CSV under `Justin/Data`. The default mix is
a starting training design, not an estimate of real equipment failure rates:

| Scenario | Percentage | Rows in 10,000 |
| --- | ---: | ---: |
| Normal | 45% | 4,500 |
| Intermittent missing readings | 15% | 1,500 |
| Persistent broken sensors | 10% | 1,000 |
| High sensor noise | 10% | 1,000 |
| Weak source signal | 10% | 1,000 |
| Different spatial decay | 5% | 500 |
| Combined difficult conditions | 5% | 500 |

Every scenario type has its own function. Change `DEFAULT_MIX` or provide a mix:

```powershell
python Justin/generate_dataset.py --mix normal=60 missing_data=15 broken_sensor=10 noisy_sensor=5 weak_signal=5 variable_spread=3 combined_stress=2
```

Percentages must total 100. Rounding preserves the requested total number of
rows. Source locations are balanced within each scenario type to within one row.

## What the scenarios mean

- Normal: noise standard deviation 0.03-0.08, random missing probability 0-5%.
- Missing data: 10%, 25%, or 50% independent missing probability, noise 0.08.
- Broken sensors: one broken sensor in 80% of groups, two in 20% on average.
  The exact same sensors stay broken for a group of 25 snapshots by default.
  Use `--failure-group-size 100` to change the duration. A new group is a separate
  simulated equipment condition, not a timed repair event. Rows are shuffled in
  the CSV; group membership remains available in `group_id`.
- Noise: Gaussian noise standard deviation 0.15-0.30. Signals remain clipped to 0-1.
- Weak signal: source intensity 0.25-0.60 instead of 1.0.
- Variable spread: distance-decay coefficient 0.25-0.45 or 0.85-1.20 instead of 0.65.
  Smaller coefficients preserve more signal at distant zones.
- Combined stress: 1-2 broken sensors, missing probability 10-50%, noise 0.15-0.25,
  source intensity 0.30-0.70, and decay 0.35-1.0.

All conditions are spatial snapshots with a single known source. There is no
time-dependent outbreak progression or outbreak-free detection class.
The numeric ranges are synthetic assumptions that can be tuned with the team.

## Training safely

- Initial inputs: the eight `zone_*_reading` columns and eight `zone_*_missing`
  columns. Blank readings need imputation fitted on the training set only.
- Target: `true_source_node`.
- Use the provided `split` column for train/validation/test. Allocation is roughly
  70/15/15 within each scenario type; whole-group rounding can change row ratios.
  Persistent-failure groups never span splits. Small datasets may omit a scenario
  from a split. Independent runs should be evaluated separately; do not merge
  same-seed datasets and randomly split duplicated samples.
- Metadata: `scenario_type`, `group_id`, `seed`, noise, missing rate, source
  intensity, decay, broken count, and `zone_*_broken` are simulation diagnostics.
  Do not automatically include them as model inputs. Broken flags reveal the
  simulator's failure cause, which the model would not ordinarily know.
- `missing_rate` applies to working sensors. Broken sensors are always missing,
  so the actual missing fraction can be higher; it is recorded separately.
- Each row includes one additional manual sample. Selection uses a missing zone
  first, otherwise a randomly chosen zone. This is a baseline sampling policy,
  not the team's eventual confidence-based adaptive policy. The independent
  manual sampler works even if the installed sensor is broken, has noise 0.04,
  and is assumed to return successfully. Original readings are preserved.
- Keep all `additional_sample_*` fields out of the initial model. For a separate
  after-sampling model, represent the selected zone and new reading explicitly
  (or replace that zone's reading and clear its missing flag), then compare on the
  same held-out rows. Only one new sample is supplied per outbreak.

Use per-scenario accuracy and graph-distance error as well as overall results.
The mixture deliberately includes difficult cases; its overall accuracy is not
a measured estimate of real-world deployment performance.
