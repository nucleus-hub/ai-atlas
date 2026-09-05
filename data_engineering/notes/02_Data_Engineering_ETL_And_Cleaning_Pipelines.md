# <span style="color:#0B3D91">Data Engineering for AI: ETL &amp; Cleaning Pipelines</span>

> Study notes covering how raw data becomes trustworthy input — the Extract → Transform → Load (ETL) process, batch vs streaming, and repeatable data-cleaning pipelines that handle missing values, noise, duplicates, and outliers.
> The "Clean" stage of the data lifecycle, made practical with hands-on pandas examples.

---

## <span style="color:#1E6FEB">Table of Contents</span>

1. [ETL Fundamentals](#1-etl-fundamentals)
2. [Data Cleaning Pipelines](#2-data-cleaning-pipelines)

---

## <span style="color:#1E6FEB">1. ETL Fundamentals</span>

### 1.1 Overview / What is it?
**ETL** stands for **Extract → Transform → Load** — the three-stage process that turns messy raw data into clean, analysis-ready data sitting in its final home. It is the backbone of nearly every data pipeline feeding an AI system.

### 1.2 Why does it matter for AI?
ETL is the machinery that implements the "**Clean**" stage of the data lifecycle. Raw data is almost never usable as-is; ETL is *how* you reliably, repeatably reshape it into something a model or a dashboard can trust. Skip it, and you are back to Garbage In → Garbage Out.

### 1.3 Key Concepts — The Three Stages

**Stage 1 — Extract** (pull raw data from its source, *exactly as it exists*):

- Pulls data out of its original source **without modifying it**, so it can be processed downstream.
- Sources: databases, flat files, APIs, logs, event streams.
- Data is pulled **as-is**: messy, inconsistent, or incomplete.
- Jobs can run on a **schedule** or be **triggered by events**.
- *Example:* a nightly job connects to a MySQL orders database and extracts all new orders from the last 24 hours into a staging file.

**Stage 2 — Transform** (clean and reshape into a usable form):

- Applies rules to **clean, standardize, and reshape** the extracted data so it is consistent and analysis-ready.
- Fix data types and standardize formats (dates, currencies).
- Remove duplicates and handle missing values.
- Join, filter, and aggregate fields to match business logic.
- *Example:* raw order dates in three different formats are standardized to `YYYY-MM-DD`, and rows with a negative amount are flagged for review.

**Stage 3 — Load** (write processed data to its destination):

- Writes the cleaned, transformed data into its **final destination** — ready for querying, reporting, or feeding AI.
- Destinations: data warehouses, databases, file stores.
- Loads can **fully replace** old data or **append incrementally**.
- Well-loaded data is what SQL and downstream AI pipelines query.
- *Example:* cleaned order records are loaded into a PostgreSQL analytics table, replacing yesterday's snapshot for the sales dashboard.

### 1.4 Batch vs Streaming ETL
Two ways to move data — chosen by *how fresh* the data must be:

| | **Batch ETL** | **Streaming ETL** |
|---|---|---|
| **How** | Scheduled chunks (nightly, hourly) | Continuously, record by record |
| **Infra** | Simpler to build, test, monitor | Needs Kafka / Kinesis |
| **Latency** | Higher — data can be hours old | Near real-time — seconds |
| **Good fit** | Daily sales reports, monthly billing | Fraud detection, live dashboards |

### 1.5 Common ETL Tools & Ecosystem
Beginners start with scripting, then adopt orchestration as pipelines grow:

| Tool | Role |
|---|---|
| **Pandas** | Python scripting for small-scale ETL |
| **Apache Airflow** | Schedules & orchestrates pipelines |
| **dbt** | SQL-based transformation layer |
| **Apache Kafka** | Streaming data pipelines |
| **Fivetran / Talend** | Managed, no-code ETL connectors |

### 1.6 Simple Example — A Raw CSV Cleaned End-to-End

**Raw input** (issues: missing name, inconsistent dates, invalid amounts):

| id | name | amount | date |
|---|---|---|---|
| 1 | A. Rao | 1200 | 2026-01-04 |
| 2 | - | | 2026/01/05 |
| 3 | b khan | 980 | 04-01-2026 |
| 4 | C. Iyer | NaN | |

**Transformed output** (clean names, standardized dates, validated numbers — ready to load):

| id | name | amount | date |
|---|---|---|---|
| 1 | A. Rao | 1200 | 2026-01-04 |
| 2 | Unknown | 0 | 2026-01-05 |
| 3 | B. Khan | 980 | 2026-01-04 |
| 4 | C. Iyer | 0 | 2026-01-04 |

### 1.7 How it works — The Code Walkthrough (pandas)

```python
import pandas as pd

df = pd.read_csv('raw_orders.csv')                    # 1. Load raw CSV
df['name'] = df['name'].fillna('Unknown')             # 2. Fill missing names
df['name'] = df['name'].str.title()                   #    standardize casing
df['date'] = pd.to_datetime(                          # 3. Parse & reformat dates
    df['date'], errors='coerce'
).dt.strftime('%Y-%m-%d')
df['amount'] = pd.to_numeric(                          # 4. Coerce amount to numeric
    df['amount'], errors='coerce'
).fillna(0)                                            #    invalid → 0
df.to_csv('clean_orders.csv', index=False)            # 5. Write cleaned CSV
```

1. Load the raw CSV into a DataFrame.
2. Fill missing names with a placeholder and standardize casing.
3. Parse and reformat every date into one consistent format.
4. Convert amount to numeric, replacing invalid values with 0.
5. Write the cleaned data back out to a new CSV.

### 1.8 Practical Example / Use Case
A retailer runs a **nightly batch ETL**: extract yesterday's orders from MySQL, transform (standardize dates/currencies, drop dupes, flag negatives), and load into a PostgreSQL analytics table that powers the morning sales dashboard. When they later add **fraud detection**, they bolt on a **streaming ETL** path via Kafka for near-real-time alerts — same concepts, different freshness requirement.

### 1.9 Key Takeaways
> - **ETL = Extract → Transform → Load**: pull raw data, clean/reshape it, write it to its final home.
> - **Extract** takes data as-is; **Transform** does the real cleaning; **Load** replaces or appends at the destination.
> - **Batch** = scheduled, simpler, higher latency; **Streaming** = continuous, real-time, needs Kafka/Kinesis.
> - Start with **pandas scripting**, graduate to **Airflow / dbt / Kafka** as pipelines scale.
> - A good transform step fixes types, dates, duplicates, and missing values — the exact issues tackled next in cleaning pipelines.

---

## <span style="color:#1E6FEB">2. Data Cleaning Pipelines</span>

### 2.1 Overview / What is it?
A **data cleaning pipeline** is a *repeatable* process for finding and fixing the recurring problems in raw data. The key word is **repeatable** — turning cleaning from a one-off script into a trustworthy, re-runnable process.

### 2.2 Why does it matter for AI?
> Repeatability turns cleaning from a one-off script into a trustworthy process.

The *same handful* of issues show up in nearly every raw dataset. If you fix them ad-hoc each time, you get inconsistent results and silent bugs. A repeatable pipeline guarantees every batch of data is cleaned the *same* way — which is exactly what a reproducible AI system needs.

### 2.3 Key Concepts — The Four Common Issues
The same problems recur in almost every dataset:

| Issue | What it is |
|---|---|
| **Missing Values** | Gaps in records that skew analysis or break model inputs entirely |
| **Noise** | Irrelevant, duplicated, or corrupted data that obscures real signal |
| **Duplicates** | The same record entered multiple times, often with slight variations |
| **Outliers** | Extreme values that distort averages and confuse models |

**Missing-value strategies** — there is no single correct fix; it depends on *how much* is missing and *why*:

- **Drop** — remove rows/columns when missingness is minimal and random.
- **Impute** — fill with mean, median, mode, or a placeholder like `'Unknown'`.
- **Flag** — add an indicator column so models can *learn the pattern* of missingness.
- *Example:* 70% of records miss a `region` field after a legacy-system merge — **flagging** this before training prevents a regionally biased model.

**Outliers & noise** — telling a genuine rare event apart from a data-entry error is the real skill:

- **Z-score** — flag values several standard deviations from the mean.
- **IQR method** — flag values far outside the interquartile range.
- **Always investigate before removing** — a real event may be your most important signal.
- *Example:* a customer age of "250" is clearly an error, but a ₹10,00,000 transaction among typical ₹1,000 purchases may be a *genuine* high-value sale.

### 2.4 How it works — The Repeatable Pipeline
Four stages, always in this order:

**Profile → Clean → Validate → Log**

| Stage | What it does |
|---|---|
| **Profile** | Understand the data first (missing %, ranges, distributions) |
| **Clean** | Apply fixes consistently |
| **Validate** | Confirm the fixes actually worked |
| **Log** | Record what changed, and why |

### 2.5 Simple Example — Before & After

**Before cleaning** (invalid ages, inconsistent casing, missing city, mixed date formats):

| age | city | signup_date |
|---|---|---|
| 250 | mumbai | 2026-01-04 |
| -5 | Delhi | 04/01/2026 |
| 34 | (blank) | 2026-01-05 |
| 29 | PUNE | 2026-01-05 |

**After cleaning** (invalid ages flagged as `NaN`, city casing standardized, dates unified):

| age | city | signup_date |
|---|---|---|
| NaN | Mumbai | 2026-01-04 |
| NaN | Delhi | 2026-01-04 |
| 34 | Unknown | 2026-01-05 |
| 29 | Pune | 2026-01-05 |

### 2.6 How it works — The Code Walkthrough (pandas)

```python
import pandas as pd
import numpy as np

df = pd.read_csv('customers_raw.csv')                  # 1. Load raw records
df.loc[                                                # 2. Validate age range
    (df['age'] < 0) | (df['age'] > 120), 'age'
] = np.nan                                             #    impossible → NaN
df['city'] = (                                         # 3. Standardize city casing
    df['city'].str.strip()
    .str.title()
    .fillna('Unknown')
)
df['signup_date'] = pd.to_datetime(                    # 4. Unify date format
    df['signup_date'], errors='coerce'
).dt.strftime('%Y-%m-%d')
df.to_csv('customers_clean.csv', index=False)          # 5. Save cleaned file
```

1. Load the raw customer records.
2. Replace impossible ages (negative or over 120) with `NaN`.
3. Strip whitespace and standardize city name casing.
4. Parse every signup date into one consistent format.
5. Save the validated, cleaned file.

### 2.7 Practical Example / Use Case
A team ingests a weekly customer export that is *always* a little different — casing varies, someone typed an age of 250, dates come in two formats. Instead of hand-fixing each week, they wrap the **Profile → Clean → Validate → Log** pipeline in a script. Every Monday it runs identically, logs what it changed, and produces a dataset they can trust — and reproduce months later during an audit.

### 2.8 Key Takeaways
> - The same four issues recur everywhere: **missing values, noise, duplicates, outliers**.
> - Missing data → choose **Drop / Impute / Flag** based on how much is missing and why (never silently turn it into zero).
> - Outliers → detect with **Z-score / IQR**, but **investigate before removing** — the outlier might be the signal.
> - A repeatable **Profile → Clean → Validate → Log** pipeline beats one-off scripts and makes cleaning trustworthy and reproducible.

---

*End of file 02 — ETL & Cleaning Pipelines complete.*
