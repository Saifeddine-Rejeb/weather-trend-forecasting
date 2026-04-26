# Weather Trend Forecasting
**PM Accelerator | Technical Assessment**


## PM Accelerator Mission

> *By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.*


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


## Project Overview

End-to-end weather forecasting pipeline built on the Global Weather Repository dataset. The notebook covers data cleaning, feature engineering, EDA, and forecasting **temperature (°C)** and **precipitation (inches)** for **Tunis, Tunisia** using an LSTM, a Random Forest, and a stacking ensemble. Predictions are interpreted using SHAP values.

The pipeline is scoped to a single location because time-series models require temporal continuity. Mixing observations from different cities into the same sequence has no physical meaning. Tunis was picked for its clear Mediterranean seasonality and variable precipitation. Changing the `location` variable at the top of the modelling section runs the same pipeline on any other city in the dataset.


## Dataset

**Source:** [Global Weather Repository on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)

Global daily weather data for capital cities worldwide, starting August 2023. Over 40 features covering temperature, wind, pressure, precipitation, humidity, visibility, air quality (CO, NO2, SO2, PM2.5, PM10), moon phase, and more. The dataset downloads automatically in the notebook via `kagglehub`.


## Repo Structure

```
weather-trend-forecasting/
├── TechAssessment.ipynb
├── GlobalWeatherRepository.csv
├── requirements.txt
└── README.md
```


## Setup & Installation

**Prerequisites:** Python 3.10+, Kaggle account with API credentials at `~/.kaggle/kaggle.json` ([setup guide](https://www.kaggle.com/docs/api))

```bash
git clone https://github.com/Saifeddine-Rejeb/weather-trend-forecasting.git
cd weather-trend-forecasting

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```


## Quick Start

```bash
jupyter notebook TechAssessment.ipynb
```

Run cells top to bottom. Everything is self-contained.


## Methodology

### Data Cleaning and Anomaly Detection

The raw dataset has no missing values and no duplicate rows. Cleaning steps:

- Dropped `last_updated_epoch`, redundant since `last_updated` is already present as a readable timestamp
- Normalised 22 country names written in non-English scripts (Arabic, Russian, Portuguese, German, French) to their English equivalents
- Audited all object columns to identify which ones encode numeric or datetime values

Outlier and anomaly analysis was performed on all numeric columns using both the **IQR rule** (1.5x IQR beyond Q1/Q3) and **Z-score** (|z| > 3). Key findings:

- `visibility_km` and `visibility_miles` have the highest IQR outlier rate (~21%) but this reflects genuine weather variability, not errors
- Air quality columns (`Carbon_Monoxide`, `Sulphur_dioxide`) contain -9999 sentinel values representing sensor failures, not real measurements
- `gust_mph` and `gust_kph` contain physically impossible values (~1800+ mph / ~2970+ kph), confirmed as sensor errors
- `precip_mm` and `precip_in` are heavily right-skewed with a median of 0, which is expected given how infrequent significant rainfall events are
- Bounded variables like `latitude`, `wind_degree`, `humidity`, `cloud`, and `moon_illumination` show zero outliers by definition

### Feature Engineering

- Parsed `last_updated` into year, month, day, weekday, and hour
- Applied **cyclical sin/cos encoding** to `hour` and `month` so the model treats 23:00 to 00:00 and December to January as continuous transitions
- Converted `sunrise`, `sunset`, `moonrise`, and `moonset` from time strings (e.g. `"04:50 AM"`) to hour-of-day integers
- MinMax-scaled all numeric features to [0, 1] before sequence creation

### Exploratory Data Analysis

- Correlation matrix across all 35+ numeric features
- Pairplot of 8 key meteorological variables: temperature, wind, pressure, precipitation, visibility, gust, CO, and NO2
- Boxplots of temperature broken down by weather condition and by month
- Boxplots of precipitation broken down by condition, month, and hour of day
- Choropleth maps of mean temperature and mean precipitation per country using GeoPandas

### Forecasting Models

All models predict the next timestep across all scaled features simultaneously (multi-output regression), using a **sliding window of 5 timesteps** as input. A single chronological 80/20 train/test split is shared across all models with no shuffling.

#### LSTM

Architecture: `LSTM(248) -> Dense(128) -> Dense(64) -> Dense(n_features)`

Trained with Adam optimizer and MSE loss for 20 epochs using a `tf.data` pipeline with shuffle and repeat on the training set. The LSTM processes the 5-step window sequentially, updating its hidden state at each step. This lets it learn from the trajectory of weather leading up to a prediction rather than just treating the window as a flat feature vector.

#### Random Forest

100 trees trained on the flattened window (`5 x n_features` input columns). The window is reshaped into a single feature vector, so the RF treats every value across all 5 timesteps as independent features. It does not learn temporal order, but handles non-linear feature interactions well and is more sample-efficient on a single city's worth of data.

### Stacking Ensemble

The scaled predictions of both base models are concatenated into a new feature matrix (`2 x n_features` columns, labelled `_lstm` and `_rf` per feature) and passed to a meta-learner trained only on training-set predictions to avoid test leakage.

Two meta-learners were evaluated:

- **Linear Regression:** learns a fixed weighted combination of both models' outputs
- **Random Forest (50 trees):** learns a non-linear, input-dependent combination

### Interpretability

Feature importances and SHAP values are computed on the RF meta-learner. The SHAP summary plot covers all `2 x n_features` stacked inputs, showing which base model's predictions the meta-learner relies on and for which targets.


## Results

Evaluated on the held-out 20% test set for Tunis:

| Model | RMSE (all) | RMSE Temp (°C) | RMSE Precip (in) |
|---|---|---|---|
| **Stack + Linear Regression** | **32.46** | **3.94** | 0.0041 |
| Stack + Random Forest | 33.55 | 5.55 | **0.0037** |
| Random Forest | 36.69 | 5.26 | 0.0038 |
| LSTM | 52.21 | 5.34 | 0.0092 |

The stacking ensemble outperforms both base models across all metrics. The Linear Regression meta-learner achieves the best temperature RMSE and the RF meta-learner achieves the best precipitation RMSE. The Random Forest outperforms the LSTM in isolation, where the RF's sample efficiency outweighs the LSTM's sequential modelling advantage at this data scale.

**Top features (RF meta-learner):** `month` is dominant by a large margin from both branches, consistent with Mediterranean seasonality. `day` and `moon_illumination` follow. `wind_direction` from the LSTM branch carries meaningful signal, likely encoding synoptic patterns like the Sirocco. The meta-learner draws more from the LSTM branch's outputs overall.


## Key Insights

- **Seasonality is the strongest signal.** Month is the top feature across all models by a wide margin.
- **Stacking consistently beats both base models.** The LSTM and RF capture different aspects of the data and combining them through a meta-learner extracts more signal than either alone.
- **Precipitation is the harder target.** It is sparse and heavily right-skewed with most observations at or near zero. A dedicated two-stage model (classify rain/no-rain, then regress the amount) would be a natural next step.
- **Unit duplicates add redundancy.** Temperature, wind, pressure, visibility, and feels-like all appear in two unit systems. Dropping one from each pair would reduce noise without losing information.


## Demo

[Link to demo video](#)


## Author

**Saifeddine Rejeb**  
PM Accelerator Technical Assessment