# 🌤 Ultimate Weather Dashboard — EDA, Forecast & Anomaly Detection

A **comprehensive weather analysis and forecasting dashboard** built with **Streamlit**, combining statistical analysis, machine learning, deep learning, and visualization techniques. This dashboard enables **exploratory data analysis**, **weather classification**, **time series modeling**, **anomaly detection**, and **animated feature visualization**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [How to Run](#how-to-run)
6. [Dashboard Components](#dashboard-components)

   * [1. Univariate Analysis](#1-univariate-analysis)
   * [2. Multivariate Analysis](#2-multivariate-analysis)
   * [3. Weather Classification Analysis](#3-weather-classification-analysis)
   * [4. Time Series ACF/PACF](#4-time-series-acfpacf)
   * [5. Clustering + PCA](#5-clustering--pca)
   * [6. Deep Learning Forecast](#6-deep-learning-forecast)
   * [7. Hybrid Model (ARIMA + LSTM Residuals)](#7-hybrid-model-arima--lstm-residuals)
   * [8. Leaderboard](#8-leaderboard)
   * [9. Anomaly Detection](#9-anomaly-detection)
   * [10. Animated Dashboard](#10-animated-dashboard)
7. [Dependencies](#dependencies)
8. [Folder Structure](#folder-structure)
9. [Notes](#notes)

---

## Project Overview

This dashboard provides a **complete end-to-end weather analysis framework**, focusing on **Seattle weather data**. It allows users to explore the dataset, detect anomalies, and forecast future temperatures using **deep learning** and **hybrid models**.

Key objectives:

* Understand weather patterns through **EDA**
* Classify weather types using numeric features
* Forecast temperature using **LSTM, GRU, BiLSTM**, and **ARIMA hybrid models**
* Detect anomalies using **IQR, Z-score, and Isolation Forest**
* Visualize data interactively, including **animated plots**

---

## Features

* **Univariate Analysis**: Histogram, boxplots, rolling mean, seasonal decomposition
* **Multivariate Analysis**: Correlation heatmaps, scatter matrix, pairwise scatter plots
* **Weather Classification**: Statistical summary per weather class and feature distribution
* **Time Series Analysis**: ACF & PACF plots for numeric features
* **Clustering**: KMeans clustering with PCA visualization
* **Forecasting**: Deep learning models (LSTM, GRU, BiLSTM) for temperature prediction
* **Hybrid Forecasting**: ARIMA + LSTM residuals for enhanced accuracy
* **Leaderboard**: Compare performance of all models (MAE, MSE)
* **Anomaly Detection**: Detect unusual weather patterns using IQR, Z-score, Isolation Forest
* **Animated Dashboard**: Live visualization of selected features, clusters, and anomalies

---

## Dataset

* **File**: `seattle-weather.csv`

* **Columns**:

  | Column        | Description                                   |
  | ------------- | --------------------------------------------- |
  | date          | Timestamp of observation                      |
  | temp_max      | Maximum temperature of the day                |
  | temp_min      | Minimum temperature of the day                |
  | precipitation | Precipitation in mm                           |
  | wind          | Wind speed                                    |
  | weather       | Weather label (e.g., rain, sun, snow, cloudy) |

* Preprocessing:

  * Dates converted to datetime and set as index
  * `weather` encoded as `weather_encoded` for numerical operations
  * Missing values forward/backward filled

---

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd ultimate-weather-dashboard
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run app.py
```

The dashboard will open in your default browser.

---

## Dashboard Components

### 1. Univariate Analysis

* Displays **histograms**, **boxplots**, **30-day rolling mean**, and **seasonal decomposition** for all numeric features.
* Helps identify **distribution patterns**, **outliers**, and **seasonal trends**.

### 2. Multivariate Analysis

* Shows **correlation heatmap** to identify feature dependencies.
* Interactive **scatter matrix** to explore feature relationships.
* **Pairwise scatter plots** (e.g., temp_max vs wind, temp_max vs precipitation) with weather labels.

### 3. Weather Classification Analysis

* Computes **statistical summary** (mean, median, std, min, max) for each numeric feature grouped by weather class.
* Boxplots per weather class show **feature distribution patterns**.

### 4. Time Series ACF/PACF

* Displays **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots for numeric features.
* Helps identify **temporal dependencies** for ARIMA modeling.

### 5. Clustering + PCA

* Performs **KMeans clustering** on numeric features.
* Uses **PCA (2 components)** to visualize clusters.
* Hover over points to see numeric feature values.

### 6. Deep Learning Forecast

* Models: **LSTM, GRU, BiLSTM**
* Uses last 30 days as input sequence to predict next day's temperature.
* Interactive chart compares **actual vs predicted** values.

### 7. Hybrid Model (ARIMA + LSTM Residuals)

* Fits an **ARIMA model** to temperature series.
* Predicts residuals using a **small LSTM network**.
* Final prediction = ARIMA forecast + LSTM residual prediction.
* Often **outperforms individual models**.

### 8. Leaderboard

* Shows **MAE and MSE** for ARIMA, Deep Learning, and Hybrid models.
* Highlights **best performing model** in a success message.

### 9. Anomaly Detection

* Detects anomalies for numeric features using:

  * **IQR (Interquartile Range)**
  * **Z-score thresholding**
  * **Isolation Forest**
* Interactive Altair chart shows anomalies in red points.

### 10. Animated Dashboard

* Live animation of a selected feature.
* Displays **clusters** and **anomalies** dynamically.
* Window size and animation speed are adjustable.

---

## Dependencies

* streamlit
* pandas
* numpy
* matplotlib
* altair
* plotly
* statsmodels
* scikit-learn
* tensorflow
* scipy

---

## Folder Structure

```
ultimate-weather-dashboard/
├── app.py
├── seattle-weather.csv
├── models/
│   ├── LSTM.keras
│   ├── GRU.keras
│   └── BiLSTM.keras
├── requirements.txt
└── README.md
```

---

## Notes

* **Caching** is used to speed up data loading and model loading.
* Seasonal decomposition might fail if the period is smaller than the number of observations; handled gracefully.
* Ensure GPU acceleration for faster deep learning training (optional).
* Animated dashboard may slow for large datasets; adjust **window size** and **speed** sliders.