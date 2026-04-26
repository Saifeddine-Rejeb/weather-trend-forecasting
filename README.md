# Weather Trend Forecasting
**PM Accelerator — Technical Assessment**

---

## PM Accelerator Mission

> *By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repo Structure](#repo-structure)
4. [Setup & Installation](#setup--installation)
5. [Quick Start](#quick-start)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Key Insights](#key-insights)
9. [Demo](#demo)

---

## Project Overview

End-to-end weather forecasting pipeline targeting **temperature (°C)** and **precipitation (inches)** for **Tunis, Tunisia**. The pipeline combines an LSTM and a Random Forest as base models, stacks their outputs through a meta-learner, and explains predictions with SHAP values.

A single location was chosen rather than training globally because time-series models rely on temporal continuity — mixing observations from different cities breaks that. Tunis was picked for its clear Mediterranean seasonality and irregular precipitation, which gives both models a meaningful signal to learn.

---

## Dataset

**Source:** [Global Weather Repository — Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)

Global daily weather data for capital cities worldwide, starting August 2023. 40+ features covering temperature, wind, pressure, precipitation, humidity, visibility, and air quality. Downloaded automatically in the notebook via `kagglehub`.

---

## Repo Structure

```
weather-trend-forecasting/
├── TechAssessment.ipynb
├── GlobalWeatherRepository.csv
├── requirements.txt
└── README.md
```

---

## Setup & Installation

**Prerequisites:** Python 3.10+, Kaggle account with API credentials at `~/.kaggle/kaggle.json` ([instructions](https://www.kaggle.com/docs/api))

```bash
git clone https://github.com/Saifeddine-Rejeb/weather-trend-forecasting.git
cd weather-trend-forecasting

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```


## Quick Start

```bash
jupyter notebook TechAssessment.ipynb
```



## Methodology

### Data Cleaning
- Dropped redundant `last_updated_epoch` column
- Normalised 22 country names from non-English scripts to English equivalents
- Identified -9999 sentinel values in air quality columns as sensor failures
- Flagged physically impossible wind gust values (~1800+ mph) as sensor errors

### Feature Engineering
- Parsed `last_updated` into year, month, day, weekday, hour
- Cyclically encoded hour and month using sin/cos — so the model knows hour 23 and hour 0 are adjacent
- Converted sunrise/sunset/moonrise/moonset time strings to hour-of-day integers
- MinMax-scaled all numeric features to [0, 1]

### EDA
- Correlation matrix
- Pairplot 
- Temperature and precipitation distributions by condition/month/hour
- Choropleth maps of global mean temperature and precipitation by country.

### Forecasting Models

All models use a **5-step sliding window** to predict the next timestep across all features, with a shared **chronological 80/20 split**.

- **LSTM** single layer (248 units) + two Dense layers. Captures how the weather *trajectory* over the past 5 steps predicts the next, rather than just the last snapshot. Trained with Adam/MSE for 20 epochs. Known limitation: no output constraints, so it can produce negative precipitation values.

- **Random Forest** 100 trees on the flattened 5-step window. Ignores temporal order but handles non-linear feature interactions well. More sample-efficient than the LSTM on a single city's worth of data, which is why it outperforms it in isolation.

### Stacking Ensemble

- Both models' scaled predictions are fed as inputs to a meta-learner (`2 × n_features` columns, labelled `_lstm` / `_rf`). The meta-learner learns when to trust each base model — and can weight them differently per target.

- Two meta-learners tested: **Linear Regression** (static blend) and **Random Forest** (input-dependent blend). Meta-model is trained on training-set predictions only to avoid leaking test data.

### Interpretability

Feature importances and SHAP values on the RF meta-learner. SHAP is preferred over raw Gini importance because it accounts for feature interactions and shows directionality.

---

## Results

| Model | RMSE (all) | RMSE Temp (°C) | RMSE Precip (in) |
|---|---|---|---|
| **Stack + Linear Regression** | **32.46** | **3.94** | 0.0041 |
| Stack + Random Forest | 33.55 | 5.55 | **0.0037** |
| Random Forest | 36.69 | 5.26 | 0.0038 |
| LSTM | 52.21 | 5.34 | 0.0092 |

The stacking ensemble beats both base models consistently. The Linear Regression meta-learner is best for temperature; the RF meta-learner is best for precipitation.

**Top features (RF meta-learner):** `month` dominates (~0.168 LSTM branch, ~0.079 RF branch), which makes sense for Mediterranean seasonality. `day`, `moon_illumination`, and `wind_direction` follow. The meta-learner leans slightly more on LSTM outputs overall.

---

## Key Insights

- **Seasonality is the strongest signal.** Month is the top feature across all models, any forecasting model for a Mediterranean city should treat the seasonal cycle as its primary input.
- **Stacking is worth it.** The Linear Regression meta-learner is trivially cheap and delivers the best temperature RMSE.
- **Precipitation is the harder target.** It is sparse and right-skewed. A dedicated two-stage model (classify rain/no-rain, then regress amount) would be a natural improvement.
- **Unit duplicates add noise.** Temperature, wind, pressure, visibility, and feels-like all appear in two units. Dropping one from each pair would clean up feature importance without losing information.

---

## Demo

[Link to demo video](#)

---

## Author

**Saifeddine Rejeb**
