# Data availability

The historical daily stock data were obtained from [Yahoo Finance](https://finance.yahoo.com/) for AAPL, JNJ, JPM, MSFT, PG, and XOM. Users should consult the [provider's data notice](https://help.yahoo.com/kb/SLN2310.html) and applicable access and licensing conditions.

The exact historical snapshot is not distributed in this supplement because redistribution rights have not been verified. Raw OHLCV, normalized observations, feature timestamps, prepared episodes and row-level predictions are not included. The MIT license covers the authored software and software documentation, not the provider's data.

The two CSV files in this repository are configuration-level study summaries. They contain aggregate scores for a ticker/fold, feature representation, target, predictor and policy. They are not price series or episode-level records.

A fresh download can differ from the snapshot used in the study. Recreating the historical fits requires authorized data for the original snapshot, complete data-preparation and partition artifacts (including information intervals), full runnable configurations, and the study's runtime environment. This package does not claim to provide those materials or an exact historical replay.
