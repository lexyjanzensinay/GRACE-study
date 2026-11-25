# GRACE: fixed-preset study supplement

GRACE is research software for cost-aware early financial time-series classification with probability calibration and reliability gates. This repository accompanies the historical fixed-predictor-preset study and is intended for source inspection, verification of saved summaries, and a synthetic runtime demonstration.

| Location | Contents |
|---|---|
| `code/` | Four unchanged study modules: estimator, temporal validation, F50 feature transformer, and experiment runner |
| `docs/configuration_records.json` | 192 descriptive records of effective model, gate and evaluation settings; inspectable, not turnkey runner inputs |
| `results/` | Saved configuration-level policy metrics and training-reference metrics; no observations or predictions per episode |
| `docs/DATA_AVAILABILITY.md` | Source, sharing limitations, and what is required for historical replay |
| `docs/STUDY_SCOPE.md` | Experiment design, interpretation, and validation boundary |
| `MANIFEST.json` | File sizes and SHA-256 values for the distributed files; the manifest does not include itself |

## Verify the distributed files and saved results

From the repository root, using Python 3.12:

```sh
python -B tools/verify_package.py
python -B tools/check_summary.py
```

Both commands use only the Python standard library. Neither trains a model or retrieves market data. The first checks package integrity and summary consistency. The second recalculates the policy means and paired outcome counts displayed in the study's fixed-preset manuscript.

## Run a synthetic demonstration

Install the recorded dependencies in an isolated Python environment, then run the original demo:

```sh
python -m pip install -r requirements-study.txt
python -B code/grace_core.py --demo --quick --n-samples 300 --seed 42
```

The demo uses generated data and is not a financial backtest or historical replay. The study runner requires separately prepared, appropriately licensed historical inputs and a complete execution configuration; neither is bundled.

## Market data and license

Historical observations were sourced from [Yahoo Finance](https://finance.yahoo.com/). The exact historical data used in the study are not redistributed here. Access is subject to the provider's conditions. See the [Yahoo Finance data notice](https://help.yahoo.com/kb/SLN2310.html) and [data availability statement](docs/DATA_AVAILABILITY.md).

The [MIT License](LICENSE) covers the authored GRACE software and its software documentation. It does not grant permission to redistribute market data. Dependencies retain their own licenses. This repository does not contain the paper or establish journal acceptance, prospective trading performance, or a guarantee of optimal stopping.
