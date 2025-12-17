# 📊 Business Insights Dashboard

An end-to-end **business intelligence and forecasting dashboard** that transforms raw transactional CSV data into actionable insights.  
The system performs **automated sentiment analysis**, **sales & profit analytics**, and **multi-model forecasting**, allowing users to compare ML algorithms directly from the UI.

---

## 🎯 Project Purpose

This project converts raw business transaction data into an interactive analytics dashboard that helps stakeholders:

- Understand **customer sentiment** from free-text reviews
- Identify **sales, profit, and regional drivers**
- Detect **recurring negative feedback trends**
- Compare **multiple forecasting models** using performance metrics
- Receive **data-driven business recommendations**

All analysis is automated from a single CSV upload.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-----|------|
| Backend | Python 3, Flask |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, NLTK |
| Visualization | Matplotlib, Seaborn (base64-encoded PNGs) |
| Frontend | HTML5, Tailwind CSS, Vanilla JavaScript |

---

## 📂 Expected CSV Schema

Uploaded CSV files **must contain all of the following columns** (order does not matter, names must match exactly):

| Column | Type | Description |
|------|------|-------------|
| `product_name` | string | Product or SKU name |
| `product_category` | string | Category identifier |
| `review` | string | Customer review text |
| `rating` | numeric | Star rating (1–5) |
| `net_profit` | numeric | Profit per transaction |
| `units_sold` | numeric | Quantity sold |
| `purchase_date` | date | Date of purchase |
| `region` | string | Sales region |

❌ Files missing any required column are rejected at upload time.

---

## 🧹 Data Cleaning & Feature Engineering

### 1. Parsing & Validation
- `purchase_date` → converted to `datetime`
- `rating` → numeric → integer
- Rows with missing dates, ratings, or reviews are dropped

### 2. Text Preprocessing
- Lowercasing
- Removal of non-alphabetic characters
- English stopword removal (NLTK)
- Stored as `cleaned_review`

### 3. Sentiment Features
- TF-IDF vectorization (`max_features=2000`, `ngram_range=(1,2)`)
- Trained once at startup using `training_data.csv`

### 4. Trend Extraction
- Separate TF-IDF on **negative reviews only**
- Extracts top complaint keywords and phrases

### 5. Time-Series Aggregations
- Monthly aggregation (`resample('ME')`) for:
  - Net profit
  - Units sold
  - Positive sentiment percentage

### 6. Business Aggregations
- Profit by category
- Sales by region
- Profit-per-unit analysis

---

## 🤖 Machine Learning Models

### 1️⃣ Sentiment Classification

| Component | Description |
|--------|-------------|
| Vectorizer | TF-IDF (2000 features, unigrams + bigrams) |
| Classifier | `LinearSVC` |
| Output | `Positive`, `Neutral`, `Negative` sentiment |

The model is trained **once at application startup** and reused for all uploads.

---

### 2️⃣ Forecasting Models (User-Selectable)

| Key | Model | Purpose |
|---|---|---|
| `linear` | Linear Regression | Baseline trend |
| `ridge` | Ridge Regression | Regularized linear model |
| `random_forest` | RandomForestRegressor | Captures nonlinear patterns |
| `gradient_boost` | GradientBoostingRegressor | Boosted ensemble |

For each model:
1. Time-series data is constructed
2. Last ~20% used as validation
3. Metrics calculated:
   - **R² Score** (goodness of fit)
   - **F1 Score** (directional accuracy: above/below median)
4. 6-month forecasts generated
5. “Next-month” KPIs surfaced in the dashboard

The frontend dropdown dynamically swaps:
- Forecast charts
- KPI cards
- R² and F1 metrics

---

### 3️⃣ Additional Analytics

- **Top Product Prediction**  
  Rolling 3-month window identifies the highest-selling product

- **Rule-Based Recommendations**
  - Focus categories (high margin, low volume)
  - Optimize cash cows (high volume, low margin)
  - Promote top products
  - Investigate low-rating products

---

## 🔄 Application Flow

1. **Upload**
   - User uploads CSV file
   - Frontend sends file to `/analyze`

2. **Backend Processing**
   - Schema validation & cleaning
   - Sentiment prediction
   - KPI aggregation
   - Chart generation (base64)
   - Forecast model execution

3. **Response Payload**
   ```json
   {
     "charts": { "...": "base64png" },
     "summary_stats": { ... },
     "trends": [ "shipping delay", "poor packaging" ],
     "recommendations": { ... },
     "forecast_models": {
       "default_key": "linear",
       "options": [...],
       "models": { ... }
     }
   }
