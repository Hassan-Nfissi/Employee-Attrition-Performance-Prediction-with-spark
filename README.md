# 📊 Employee Attrition & Performance Prediction with Apache Spark

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![PySpark](https://img.shields.io/badge/PySpark-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://spark.apache.org/docs/latest/api/python/)
[![MLlib](https://img.shields.io/badge/Spark%20MLlib-Machine%20Learning-FF6F00?style=for-the-badge)](https://spark.apache.org/mllib/)
[![Structured Streaming](https://img.shields.io/badge/Spark-Structured%20Streaming-00599C?style=for-the-badge)](https://spark.apache.org/streaming/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end Big Data Machine Learning system built with **Apache Spark (PySpark)** to predict employee attrition risk both in **offline batch mode** and **real-time streaming mode**. 

This repository demonstrates how to build scalable ML pipelines, share modular feature-engineering logic, serialize models, and consume live data streams via Spark Structured Streaming over TCP Sockets.

---

## 🎯 Project Overview

Employee turnover (attrition) is a critical challenge for HR departments. Identifying employees at risk of leaving allows organizations to take proactive retention measures. 

This project leverages the **IBM HR Analytics Employee Attrition & Performance** dataset to:
1. Clean and engineer domain-specific features using PySpark SQL.
2. Train a **Logistic Regression classification pipeline** via PySpark ML.
3. Export and persist the trained `PipelineModel`.
4. Deploy a **Spark Structured Streaming** application to process live HR event streams in real time.

---

## 🏗️ System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │ IBM HR Dataset (.csv)                        │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │   preprocessor.py            │
                          │ (Shared Feature Engineering) │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │         train.py             │
                          │   PySpark MLlib Pipeline     │
                          └──────────────┬───────────────┘
                                         │ (Model Serialization)
                                         ▼
                          ┌──────────────────────────────┐
                          │   HRAttrition_Model /        │
                          └──────────────┬───────────────┘
                                         │
 ┌───────────────────────────┐           │ (Model Loading)
 │ TCP Socket Stream (Data)  ├───────────┼──────────────────────┐
 └───────────────────────────┘           ▼                      │
                          ┌──────────────────────────────┐      │
                          │         stream.py            │      │
                          │ Spark Structured Streaming   │◄─────┘
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │ Live Real-Time Predictions   │
                          └──────────────────────────────┘
```

---

## ✨ Key Features

- ⚙️ **Modular Preprocessing Engine (`preprocessor.py`):** DRY architecture where feature engineering rules, binary encoding, and thresholding logic are encapsulated in a single module shared across batch and streaming pipelines.
- 🤖 **PySpark ML Pipeline (`train.py`):** Automatically indexes categorical variables (`StringIndexer`), applies one-hot encoding (`OneHotEncoder`), vectorizes features (`VectorAssembler`), and trains a `LogisticRegression` model.
- ⚡ **Real-Time Streaming (`stream.py`):** Consumes streaming socket data using PySpark Structured Streaming (`spark.readStream`), parses incoming records dynamically, casts data types, applies transformations, and outputs real-time predictions (`prediction` and `probability`).
- 🔄 **Feature Engineering & Business Logic:**
  - Aggregates overall satisfaction scores from 5 key indicators (*Environment, Job Involvement, Job Satisfaction, Relationship, Work-Life Balance*).
  - Encodings and threshold transformations for numeric & categorical features (*Age, DistanceFromHome, MonthlyIncome, OverTime, TotalWorkingYears*, etc.).

---

## 📁 Repository Structure

```
Employee-Attrition-Performance-Prediction-with-spark/
│
├── datasets/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # IBM HR Analytics Dataset
│
├── HRAttrition_Model/                           # Serialized Spark ML PipelineModel
│   ├── metadata/
│   └── stages/
│
├── preprocessor.py                              # Shared feature engineering & transformations module
├── train.py                                     # Offline batch ML training & pipeline serialization
├── stream.py                                    # Spark Structured Streaming real-time prediction
└── README.md                                    # Comprehensive project documentation
```

---

## 🛠️ Tech Stack & Requirements

- **Engine:** Apache Spark 3.x
- **Language:** Python 3.8+ / PySpark
- **ML Framework:** PySpark MLlib (`Pipeline`, `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `LogisticRegression`)
- **Streaming Engine:** Spark Structured Streaming (`readStream`, `writeStream`, TCP Sockets)
- **Data Source:** IBM HR Analytics Employee Attrition & Performance Dataset

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.8+ and Java 8 or 11 are installed on your machine. Install PySpark via pip:

```bash
pip install pyspark
```

### 2. Clone the Repository
```bash
git clone https://github.com/Hassan-Nfissi/Employee-Attrition-Performance-Prediction-with-spark.git
cd Employee-Attrition-Performance-Prediction-with-spark
```

---

## 💻 How to Run

### Step 1: Train the Machine Learning Model
Run `train.py` to process the dataset, train the PySpark ML pipeline, and save the serialized model:

```bash
python train.py
# OR using spark-submit
spark-submit train.py
```
*Outputs a trained `HRAttrition_Model` directory containing all pipeline stages.*

---

### Step 2: Run Real-Time Streaming Predictions

1. **Start a TCP Socket Server (e.g., via Netcat):**
   ```bash
   nc -lk 9999
   ```

2. **Launch the Streaming Engine:**
   Update the host/port in `stream.py` if necessary, then run:
   ```bash
   python stream.py
   # OR using spark-submit
   spark-submit stream.py
   ```

3. **Stream Sample CSV Records:**
   Paste sample CSV rows into your Netcat terminal. Spark Structured Streaming will ingest, transform, and output real-time attrition predictions directly to the console!

---

## 📊 Feature Engineering Rules Reference

| Feature | Transformation Rule |
| :--- | :--- |
| **OverTime / Gender** | Encoded as Binary (1 for `Yes`/`Female`, 0 otherwise) |
| **Total Satisfaction** | Mean score of 5 satisfaction fields; converted to boolean (`>= 2.8`) |
| **Age** | Boolean threshold (`< 35` years) |
| **Monthly Income** | Boolean threshold (`< 4000`) |
| **Distance From Home** | Boolean threshold (`> 10` miles/km) |
| **Total Working Years**| Boolean threshold (`< 8` years) |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Hassan Nfissi**  
*Software Engineering, Big Data & DevOps*  
- **GitHub:** [@Hassan-Nfissi](https://github.com/Hassan-Nfissi)
- **LinkedIn:** [Hassan Nfissi](https://www.linkedin.com/in/hassan-nfissi-9b784428b)
