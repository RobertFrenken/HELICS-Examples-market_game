Advanced Orchestration Example
==============================

This example runs repeated EV charging simulations and stores CSV outputs under
`results/`. The peak-power summary plot is generated from those saved CSV files.

Generate the plot from the included sample results:

```bash
python3 plot_samples.py --no-show
```

That writes:

```text
montecarlo-ev-peak-power.png
```

To write the image somewhere else:

```bash
python3 plot_samples.py 30 . --output-file /tmp/montecarlo-ev-peak-power.png --no-show
```

The PNG is generated output and is ignored by the repository-wide `*.png` rule.
The reusable inputs are the CSV files under `results/`.
