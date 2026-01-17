
# Madrid Air Quality Forecasting with LSTM

## Overview
This repository contains a Jupyter notebook (`Homework_2.ipynb`) for a deep learning project focused on time series forecasting of air quality in Madrid. The model uses LSTM (Long Short-Term Memory) networks implemented in PyTorch to predict pollutants like NO₂, O₃, PM10, and PM2.5 based on historical data from 2001-2018.

The dataset is sourced from Kaggle: [Air Quality in Madrid (2001-2018)](https://www.kaggle.com/datasets/decide-soluciones/air-quality-madrid). Training data spans 2001-2017, with 2018 used for testing. The project includes core tasks such as data loading, model implementation, performance evaluation (e.g., MSE, MAE), and model weight saving.

### Key Features
- **Core Implementation**: LSTM model for sequence forecasting. Uses PyTorch's built-in LSTM cell.
- **Data Handling**: Automatic Kaggle API setup in Colab for dataset download. Preprocessing includes normalization and handling missing values.
- **Evaluation**: Metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE) on test data. Visualizations of predictions vs. actuals.
- **Model Weights**: Saved as `model_weights.pth` (or similar; check notebook for exact file).
- **Extra Feature 1**: [Describe briefly, e.g., Integration of meteorological data for improved forecasting accuracy.]
- **Extra Feature 2**: Station clustering using KMeans and DBSCAN to group monitoring stations by pollutant profiles (e.g., urban vs. suburban patterns). Observations: KMeans identifies 3 clusters (high NO₂/low O₃ for traffic areas; high O₃ for suburbs; balanced for mixed). DBSCAN detects outliers like industrial stations.

## Requirements
- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib, Scikit-learn
- Kaggle API (for dataset download)

Install dependencies:
```
pip install torch numpy pandas matplotlib scikit-learn kaggle
```

## Setup and Usage
1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/Madrid-Air-Quality-Forecasting-LSTM.git
   cd Madrid-Air-Quality-Forecasting-LSTM
   ```

2. **Dataset**:
   - The notebook handles automatic download via Kaggle API. You'll need a Kaggle account and API token (instructions in notebook).
   - Alternatively, manually download the dataset from the link above and place CSVs in `csvs_per_year/`.

3. **Run the Notebook**:
   - Open `notebook.ipynb` in Jupyter or Google Colab.
   - Execute cells sequentially. It includes setup, data loading, training, evaluation, and extensions.
   - For GPU acceleration: Ensure CUDA is available (notebook checks for it).

4. **Training and Prediction**:
   - Train the model on partial/full data (configurable).
   - Evaluate on 2018 test set.
   - Load saved weights for inference: `model.load_state_dict(torch.load('model_weights.pth'))`.

## Results
- Pollutant sample counts: NO₂ (high volume), O₃, PM10, PM2.5.
- Model performance: [Briefly mention if available, e.g., Low MSE on NO₂ forecasts; visualizations in notebook.]
- Clustering insights: Reveals spatial pollution differences (e.g., traffic-dominated vs. photochemical).

## License
MIT License. Feel free to use and modify.

## Acknowledgments
- Dataset by Decide Soluciones via Kaggle.
- Project inspired by academic homework on time series forecasting.