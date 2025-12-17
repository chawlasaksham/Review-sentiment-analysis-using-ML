# Business Insights Dashboard – Technical Overview

## 1. Purpose

This project turns raw transactional CSV files into an interactive business-intelligence dashboard. It automates customer review sentiment analysis, highlights sales/ profit drivers, and provides multi‑model forecasts with performance metrics so stakeholders can compare algorithms directly inside the UI.

---

## 2. Tech Stack

| Layer | Tools |
| --- | --- |
| Backend | Python 3, Flask |
| Data / ML | pandas, NumPy, scikit-learn, NLTK |
| Visualization | Matplotlib, Seaborn (rendered as base64 PNGs) |
| Frontend | HTML5, Tailwind CSS, vanilla JavaScript |

---

## 3. Expected CSV Schema

Upload files **must** contain the following columns (any order, header names must match exactly):

| Column | Type | Description |
| --- | --- | --- |
| `product_name` | string | SKU or product label |
| `product_category` | string | Category identifier |
| `review` | string | Free‑text customer feedback |
| `rating` | int/float | 1–5 star rating |
| `net_profit` | numeric | Profit for the row (currency units) |
| `units_sold` | numeric | Quantity sold |
| `purchase_date` | date | Parsable date string; auto-resampled monthly |
| `region` | string | Sales region / market |

Rows missing any of these columns are rejected at upload time.

---

## 4. Data Cleaning & Feature Engineering

1. **Parsing & Validation**
   - `purchase_date` → `datetime64`, coercing invalid entries to `NaT` (later dropped).
   - `rating` → numeric then cast to `int`.
   - Rows without valid `purchase_date`, `rating`, or `review` are discarded.

2. **Text Normalization (`clean_text`)**
   - Lowercase.
   - Strip non A–Z characters.
   - Remove English stopwords via NLTK.
   - Output stored in `cleaned_review`.

3. **Sentiment Feature Matrix**
   - TF‑IDF (`max_features=2000`, `ngram_range=(1,2)`) fitted on `training_data.csv`.
   - Linear SVC is trained once at startup via `train_model()` and reused for every upload.

4. **Negative Trend Extraction**
   - TF‑IDF (top 10 unigrams/bigrams) on only negative reviews per upload to surface recurring complaints.

5. **Time-Series Aggregations**
   - Monthly (`resample('ME')`) sums for `net_profit` and `units_sold`.
   - Positive-sentiment share per month for trend forecasting.

6. **Category & Region Metrics**
   - `groupby` aggregations to compute profit by category, units sold per region, profit-per-unit, etc.

---

## 5. Machine Learning Models

### 5.1 Sentiment Classification
| Component | Details |
| --- | --- |
| Vectorizer | TF‑IDF, 2000 features, 1–2 grams |
| Classifier | `LinearSVC(random_state=42, dual=True, max_iter=2000)` |
| Output | `predicted_sentiment` column with {Positive, Neutral, Negative} |

### 5.2 Forecasting Suite (switchable via dashboard dropdown)

| Key | Model | Usage |
| --- | --- | --- |
| `linear` | `LinearRegression()` | Baseline trend line |
| `ridge` | `Ridge(alpha=1.0)` | Regularized linear fit |
| `random_forest` | `RandomForestRegressor(n_estimators=300, random_state=42)` | Captures nonlinearities |
| `gradient_boost` | `GradientBoostingRegressor(learning_rate=0.05, n_estimators=400, random_state=42)` | Boosted ensemble for smaller datasets |

For each model the backend:
1. Builds time-indexed series for `net_profit`, `units_sold`, positive-sentiment percentage, and the top-selling category.
2. Splits the last ~20 % of observations for evaluation.
3. Trains the selected regressor, predicts the holdout portion, and computes:
   - **R²** – goodness of fit.
   - **F1 score** – converts actual vs. predicted values into “above/below median” classes to measure directional accuracy.
4. Generates 6‑month forecasts plus “next month” headline metrics.

The frontend dropdown swaps all four forecast charts plus the KPI cards and surfaces the R²/F1 for that model.

### 5.3 Ancillary Analytics
- **Top product prediction** – rolling 3‑month aggregation, picks the SKU with maximum units sold.
- **Recommendations** – rule-based analysis on profit-per-unit & sales volume to produce “Focus”, “Optimize”, “Promote”, “Investigate” advice cards.

---

## 6. Application Flow

1. **Upload** – user selects CSV; JS posts to `/analyze`.
2. **Backend Processing**
   - Validates schema & cleans rows.
   - Runs sentiment inference on each review.
   - Aggregates KPIs, extracts negative-review trends, assembles Matplotlib/Seaborn charts (encoded as base64).
   - Runs `generate_model_results()` to build the multi-model forecast payload.
3. **Response JSON**
   ```json
   {
     "charts": { "sentiment_pie": "...", "profit_forecast": "...", ... },
     "summary_stats": { ... , "predictions": {...} },
     "trends": ["shipping delay", ...],
     "recommendations": {...},
     "forecast_models": {
       "default_key": "linear",
       "options": [{ "key": "linear", "label": "Linear Regression" }, ...],
       "models": {
         "linear": {
           "charts": {...},
           "predictions": {...},
           "metrics": {"r2": 0.87, "f1": 0.75}
         },
         ...
       }
     }
   }
   ```
4. **Frontend Rendering**
   - Base64 charts injected into `<img>` tags.
   - Stats/ cards populated with formatted strings.
   - Dropdown populated from `forecast_models.options`; changing selection calls `applyModelSelection()` to hot-swap charts and metrics.

---

## 7. Running the Project

```bash
cd "/Users/sakshamchawla/Desktop/3rd sem github dump/os project"
source ~/envs/os-project/bin/activate   # or python3 -m venv venv && source venv/bin/activate
pip install -r requirement.txt
python app.py     # launches on http://127.0.0.1:8000
```

Upload `sales_data.csv` (or any correctly formatted file) to view the analytics dashboard.

---

## 8. Key Deliverables

- Automated sentiment classification for every review.
- Trend extraction focused on negative feedback.
- Rich KPI visualizations (profit over time, sales by region, category performance, rating vs. sentiment heatmap).
- Forecast area with interchangeable ML models + live R²/F1 scoring.
- Rule-based recommendations and top-product prediction for actionability.

This document summarizes the core functionality, preprocessing pipeline, and modeling choices so future contributors or evaluators can quickly understand the system end-to-end.


