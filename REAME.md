# Pollution LSTM Forecaster

Time-series LSTM pipeline for predicting pollution levels from the Beijing PM2.5 dataset.

Based on the work found here: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
and the dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
## Files
- `main.py`: Main script to run the pipeline.
- `dataprocessing.py`: Data cleaning, normalization, and sequence creation.
- `model.py`: LSTM model creation, training, and evaluation.
- `functions.py`: Utility functions for plotting and reporting.
- `config.py`: Configuration parameters for file paths, model, and training.
- `silenceStdError.py`: Suppresses TensorFlow warnings.
- `requirements.txt`: Required Python packages.

## Setup
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`

## Workflow
1. **Data prep** (`dataprocessing.load_clean_save`):
   - Cleans NaNs, one-hot encodes `wnd_dir`, saves processed and normalized CSV files.
2. **Sequence creation** (`dataprocessing.create_sequences`):
   - Sliding window of length `config.SEQUENCE_LENGTH` with pollution as target.
3. **Training** (`main.py`):
   - Splits into 70/15/15 train/val/test sets.
   - Builds LSTM from `model.create_lstm_model`, trains with early stopping.
4. **Evaluation & reporting**:
   - `model.evaluate_model` prints MAE/MSE/R², plots actual vs predicted, supports saving CSV and threshold-based classification metrics (`functions.reg_classification_report`).
   - Additional utilities in `functions.py` for plotting and data inspection.

## Key Config (`config.py`)
- Files: `ppm_raw.csv`, `ppm_cleaned.csv`, `ppm_normalized.csv`, `ppm_actual_predicted.csv`
- Model: `LSTM_UNITS=10`, `DENSE1_UNITS=5`, `DENSE2_UNITS=2` (optional)
- Training: `SEQUENCE_LENGTH=5`, `BATCH_SIZE=32`, `EPOCHS=100`, `PATIENCE=20`
- Classification threshold: `0.30` relative error

## Running
```bash
python main.py