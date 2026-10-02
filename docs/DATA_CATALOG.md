# FinSight Data Catalog

## Overview
- **Total Files Scanned:** 155
- **Total Raw Data Size:** 8.56 GB
- **Primary Categories:** Macro Interest Rates, RBI Scheduled Commercial Bank Returns, Financial NLP Sentiment, Hybrid Financial QA (TAT-QA), Credit Risk & Anomaly Detection, Global Events (GDELT).

## Component Mapping Table

| Component | Sub-Module | Local Dataset Source | Role in System | Fallback / Enrichment Strategy |
|---|---|---|---|---|
| **Sentiment Agent** | FinBERT Fine-tuning & Scoring | `Datasets/sent_train.csv`, `Datasets/train.csv` | Labeled financial headline & tweet sentiment (positive/negative/neutral) | Cached local historical news & RSS feed |
| **Filings-RAG Agent** | Hybrid RAG & Financial QA | `Datasets/tatqa_dataset_train.json`, `Datasets/financebench_document_information.jsonl` | 2,200+ financial table & text context pairs with structured questions/answers | Local annual reports (Tata Motors, Reliance, etc.) + ChromaDB/BM25 |
| **Quant Agent** | Credit Risk / Financial Health | `Datasets/statlog+german+credit+data/` | 1,000 corporate/consumer credit records with 20 financial risk attributes | Altman Z-score & Piotroski F-score rule-plus-model scorecards |
| **Quant Agent** | Anomaly Detection | `Datasets/creditcard.csv`, `Datasets/PS_...log.csv` | Multi-dimensional financial anomaly & transaction deviation benchmarks | Isolation Forest + Rolling MAD on equity price/volume time-series |
| **Quant Agent** | Risk-Free Benchmark & Macro | `Datasets/SOFR180DAYAVG.csv`, `Datasets/SOFR90DAYAVG.csv`, `Datasets/AMBOR*.csv` | Benchmark interest rates & discount factor references | RBI Repo rate & 10Y Indian Sovereign Yield (6.5% base) |
| **Data Agent** | Sector & Banking Macro Context | `Datasets/Basic Statistical Returns of SCBs - Vol 52-54/`, `Datasets/India.xls` | Sector-wise credit growth, occupation-wise loan distribution, bank interest rates | Offline processed cache of historical NSE/BSE stock prices & fundamentals (Tata Motors + peers) |
| **Classical ML & DL** | Direction Classifier & Forecaster | Local Processed Cache (`data/processed/`) | Daily OHLCV, volume, technical indicators, lagged returns, financial ratios | Pre-cached 5+ year historical daily price series for Tata Motors (`TATAMOTORS.NS`), NIFTY50, and sector peers |

## Detailed Dataset Inventory

| Dataset Path | Format | Size | Rows | Cols | Columns / Fields | Target Component | Purpose & Description |
|---|---|---|---|---|---|---|---|
| `Datasets/GDELT.MASTERREDUCEDV2.TXT` | txt | 6276.52 MB | None | None |  | Sentiment Agent  | Global events and macro geopolitical news dataset |
| `Datasets/train.jsonl` | jsonl | 1484.92 MB | > 1,000,000 | 23 | filename, cik, year, section_1 (+19 more) | Sentiment Agent  | Financial sentiment classification & headline scoring |
| `Datasets/PS_20174392719_1491204439457_log.csv` | csv | 470.67 MB | > 250,000 | 11 | step, type, amount, nameOrig (+7 more) | Quant Agent  | Financial transaction anomaly & fraud detection |
| `Datasets/creditcard.csv` | csv | 143.84 MB | > 250,000 | 31 | Time, V1, V2, V3 (+27 more) | Quant Agent  | Financial transaction anomaly & fraud detection |
| `Datasets/train (1).json` | json | 74.59 MB | None | None |  | Data Agent & Feature Store | Financial research dataset |
| `Datasets/train (2).json` | json | 74.59 MB | None | None |  | Data Agent & Feature Store | Financial research dataset |
| `Datasets/train (3).json` | json | 74.59 MB | None | None |  | Data Agent & Feature Store | Financial research dataset |
| `Datasets/train (4).json` | json | 74.59 MB | None | None |  | Data Agent & Feature Store | Financial research dataset |
| `Datasets/train.json` | json | 74.59 MB | None | None |  | Sentiment Agent  | Financial sentiment classification & headline scoring |
| `Datasets/tatqa_dataset_train.json` | json | 12.25 MB | 2201 | 3 | table, paragraphs, questions | Sentiment Agent  | Financial sentiment classification & headline scoring |
| `Datasets/sent_train.csv` | csv | 838.52 KB | 9938 | 2 | text, label | Sentiment Agent  | Financial sentiment classification & headline scoring |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 3/Table No 3.9 District-wise classification of outstanding credit of scheduled commercial banks according to occupation.xlsx` | xlsx | 500.54 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 53/Section 3/Table No 3.9 District-wise classification of outstanding credit of scheduled commercial banks according to occupation.xlsx` | xlsx | 498.45 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 52/Table No 3.9.xlsx` | xlsx | 483.89 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 3/Table No 3.6 State and bank group-wise classification of outstanding credit of scheduled commercial banks according to occupation.xlsx` | xlsx | 337.78 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 53/Section 3/Table No 3.6 State and bank group-wise classification of outstanding credit of scheduled commercial banks according to occupation.xlsx` | xlsx | 335.74 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 52/Table No 3.6.xlsx` | xlsx | 333.84 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 3/Table No 3.7 State and Population group-wise classification of outstanding credit of scheduled commercial banks according to occupation.xlsx` | xlsx | 247.23 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 53/Section 3/Table No 3.7 State and Population group-wise classification of outstanding credit of scheduled commercial banks according to occupation.xlsx` | xlsx | 245.64 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 52/Table No 3.7.xlsx` | xlsx | 244.32 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/train.csv` | csv | 157.60 KB | 961 | 7 | sentence, snippets, target, sentiment_score (+3 more) | Sentiment Agent  | Financial sentiment classification & headline scoring |
| `Datasets/statlog+german+credit+data/german.data-numeric` | data-numeric | 99.61 KB | None | None |  | Quant Agent  | Credit risk scoring, loan default risk & financial health scorecard |
| `Datasets/Basic Statistical Returns of SCBs - Vol 52/Table No 2.14.xlsx` | xlsx | 93.40 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 2/Table No 2.14 District and Population Group-wise Credit of SCBs According to Place of Sanction.xlsx` | xlsx | 93.23 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/financebench_document_information.jsonl` | jsonl | 86.70 KB | 361 | 6 | doc_name, company, gics_sector, doc_type (+2 more) | Filings-RAG Agent & Benchmark evaluation (Section 4.4) | SEC 10-K filings metadata and retrieval evaluation set |
| `Datasets/Basic Statistical Returns of SCBs - Vol 53/Section 2/Table No 2.14 District and Population Group-wise Credit of SCBs According to Place of Sanction.xlsx` | xlsx | 79.08 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/statlog+german+credit+data/german.data` | data | 77.92 KB | None | None |  | Quant Agent  | Credit risk scoring, loan default risk & financial health scorecard |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 3/Table No 3.5 Population Group and Bank Group-wise Classification of Outstanding Credit of SCBs According to Occupation.xlsx` | xlsx | 60.76 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 53/Section 3/Table No 3.5 Population Group and Bank Group-wise Classification of Outstanding Credit of SCBs According to Occupation.xlsx` | xlsx | 60.63 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 52/Table No 3.5.xlsx` | xlsx | 60.24 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/India.xls` | xls | 56.20 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 3/R568_ExcelTemplate.xlsx` | xlsx | 50.60 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 53/Section 3/Table No 3.8 State and population group-wise classification of outstanding credit of small borrowal accounts of scheduled commercial banks according to occupation.xlsx` | xlsx | 47.75 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 52/Table No 3.8.xlsx` | xlsx | 47.75 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 3/Table No 3.8 State and population group-wise classification of outstanding credit of small borrowal accounts of scheduled commercial banks according to occupation.xlsx` | xlsx | 47.66 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/SOFR180DAYAVG.csv` | csv | 39.68 KB | 1305 | 2 | observation_date, SOFR180DAYAVG | Quant Agent  | Macroeconomic interest rates & risk-free benchmark series |
| `Datasets/SOFR90DAYAVG.csv` | csv | 39.68 KB | 1305 | 2 | observation_date, SOFR90DAYAVG | Quant Agent  | Macroeconomic interest rates & risk-free benchmark series |
| `Datasets/Basic Statistical Returns of SCBs - Vol 54/Section 3/Table No 3.4 Size of Credit Limit-wise Classification of Outstanding Credit of SCBs According to Occupation.xlsx` | xlsx | 31.49 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 53/Section 3/Table No 3.4 Size of Credit Limit-wise Classification of Outstanding Credit of SCBs According to Occupation.xlsx` | xlsx | 31.34 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |
| `Datasets/Basic Statistical Returns of SCBs - Vol 52/Table No 3.4.xlsx` | xlsx | 31.27 KB | Variable (Spreadsheet) | None |  | Data Agent  | RBI banking system sector-wise credit, deposit and interest rate distribution |

*(Note: 100+ RBI SCB statistical volume tables are summarized under `Datasets/Basic Statistical Returns of SCBs`)*
