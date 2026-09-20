# YouTube Trending Video Statistics — India EDA

A reproducible Exploratory Data Analysis (EDA) project using the India portion of the Kaggle **Trending YouTube Video Statistics** dataset.

## Project objective

The project demonstrates an end-to-end data-analysis workflow:

1. Load and understand the dataset
2. Inspect structure and data quality
3. Handle exact duplicate rows and inspect missing values
4. Parse date/time fields
5. Perform quantitative analysis
6. Explore numerical, categorical, channel and temporal patterns
7. Study relationships between engagement variables
8. Create visualizations
9. Document evidence-based findings and limitations

## Dataset

- File: `INvideos.csv`
- Region: India
- Initial size: **37,352 rows × 16 columns**
- Missing values in initial inspection: **561 in `description`**
- Exact duplicate rows in initial inspection: **4,263**
- Analysis copy after exact-duplicate removal: **33,089 rows**

The original dataset is available from Kaggle under the Trending YouTube Video Statistics dataset by `datasnaek`.

## Repository structure

```text
youtube-trending-eda/
├── data/
│   ├── INvideos.csv
│   ├── IN_category_id.json
│   └── data_dictionary.csv
├── notebooks/
│   └── YouTube_Trending_EDA.ipynb
├── outputs/
│   ├── key_metrics.csv
│   ├── engagement_describe.csv
│   ├── category_summary.csv
│   ├── channel_summary.csv
│   ├── engagement_correlation.csv
│   ├── daily_trending_summary.csv
│   ├── findings.txt
│   └── *.png
├── eda.py
├── requirements.txt
└── README.md
```

## Main findings from the current run

- Views range from **4,024** to **125,432,237**; median views are **275,027**.
- Views and likes have a Pearson correlation of approximately **0.853** in the cleaned analysis copy.
- Views and comment count have a Pearson correlation of approximately **0.698**.
- The cleaned analysis copy contains **17 category labels** and **16,307 unique video IDs**.
- Parsed trending dates range from **2017-11-14 to 2018-06-14**.

These are descriptive observations from this dataset and should not be interpreted as causal relationships or as representative of all YouTube content.

## Run locally

```bash
pip install -r requirements.txt
python eda.py
```

Then open `notebooks/YouTube_Trending_EDA.ipynb` in Jupyter or VS Code.
