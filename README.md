<div align="center">

# 🎧 SupportIQ
### Intelligent Customer Support Query Classification Using Machine Learning

**An NLP-powered system for automatically categorizing customer support requests.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Status](https://img.shields.io/badge/Status-ML%20Prototype-yellow)

</div>

---

## 📌 Overview

**SupportIQ** is a machine learning project designed to help organize customer support requests by predicting the category of an incoming customer query. It uses **TF-IDF** text features and a **Logistic Regression** classifier to learn patterns in labeled support messages.

Instead of manually sorting every request, a support team could use the classifier as a starting point for ticket routing, prioritization, or downstream automation.

> **Current scope:** The code in this repository trains, evaluates, and saves a query classification model. Automatic response generation and a deployed support application are not included in the current repository.

## ✨ Features

- **Text classification:** Predicts a support category from a customer message.
- **NLP feature extraction:** TF-IDF with unigrams and bigrams, English stop-word removal, and sublinear term frequency.
- **Class imbalance handling:** Logistic Regression with balanced class weights.
- **Reproducible evaluation:** Stratified 80/20 train-test split with a fixed random seed.
- **Performance reporting:** Accuracy and per-class precision, recall, and F1-score.
- **Reusable artifacts:** Saves the trained classifier, TF-IDF vectorizer, and class labels with Joblib.

## 🧠 How It Works

```text
Labeled customer queries (CSV)
              │
              ▼
   Remove missing query/label rows
              │
              ▼
   Stratified train/test split
              │
              ▼
      TF-IDF vectorization
      (1–2 grams, max 10k features)
              │
              ▼
       Logistic Regression
      (balanced class weights)
              │
              ▼
       Test-set evaluation
              │
              ▼
   Saved model + vectorizer + labels
```

### Model configuration

| Component | Implementation |
| --- | --- |
| Task | Supervised multiclass text classification |
| Input | Customer support query text |
| Target | Category label |
| Vectorizer | `TfidfVectorizer` |
| N-grams | `(1, 2)` |
| Maximum vocabulary features | `10,000` |
| Stop words | English |
| TF scaling | `sublinear_tf=True` |
| Classifier | `LogisticRegression` |
| Class weights | `balanced` |
| Maximum iterations | `1000` |
| Train/test split | `80% / 20%` |
| Random seed | `42` |
| Evaluation | Accuracy and classification report |

## 🗂️ Repository Structure

```text
Customer-Support-System/
├── data/
│   └── dataset.csv           # Labeled customer queries
├── model/                    # Serialized artifacts / output directory
│   ├── model.pkl             # Generated classifier
│   ├── vectorizer.pkl        # Generated TF-IDF vectorizer
│   └── labels.pkl            # Generated category labels
├── train_model.py            # Training and evaluation script
└── requirements.txt         # Python dependencies
```

> The three `.pkl` files are produced by running the training script. Their presence in a fresh clone depends on whether they have been committed to the repository.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- `pip`
- A labeled CSV file at `data/dataset.csv`

### 1. Clone the repository

```bash
git clone https://github.com/GSgurnoorsingh/Customer-Support-System.git
cd Customer-Support-System
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

The training script expects a CSV at `data/dataset.csv` containing at least these columns:

| Column | Description |
| --- | --- |
| `query` | Customer's written support request |
| `category` | Correct category assigned to the request |

Example format (illustrative only; not a sample from the actual dataset):

```csv
query,category
"I cannot sign in to my account",account_access
"Where is my delivery?",shipping
"I want a refund",refund
```

Each category needs enough labeled examples to support a stratified train/test split. Rows missing either required field are dropped during training.

### 5. Train and evaluate the model

```bash
python train_model.py
```

The script prints the dataset size, available categories, model accuracy, and a per-class classification report. It then saves:

```text
model/model.pkl
model/vectorizer.pkl
model/labels.pkl
```

## 🔍 Using the Trained Model

After running the training script, you can load the artifacts in a Python session:

```python
import joblib

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

query = ["I need help accessing my account"]
features = vectorizer.transform(query)
predicted_category = model.predict(features)[0]

print("Predicted category:", predicted_category)
```

> Only load Joblib files from sources you trust. Serialized Python objects can execute code when loaded.

## 📊 Model Evaluation

The training script calculates **accuracy** and a **classification report** (precision, recall, F1-score, and support for each category) on a held-out test set.

To document a verified result, run `python train_model.py` and add the metrics below:

| Metric | Test-set result |
| --- | --- |
| Accuracy | _Add output from your run_ |
| Macro F1-score | _Add output from your run_ |
| Weighted F1-score | _Add output from your run_ |

**Evaluation notes:** A single stratified split provides a useful baseline, but cross-validation and testing on new, real-world customer messages would give a more robust picture of generalization.

## 🛠️ Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Core implementation |
| pandas | CSV loading and data preparation |
| scikit-learn | Text vectorization, classification, and evaluation |
| Joblib | Saving and loading model artifacts |

The repository's `requirements.txt` also lists NumPy and Streamlit. The currently published training script does not use Streamlit as an interface.

## 🧩 Current Limitations

- The model predicts **categories**, not customer-facing answers.
- TF-IDF depends on vocabulary observed during training and may struggle with unfamiliar phrasing.
- Model performance may change with new categories or a different distribution of customer messages.
- The published training script uses a single train/test split rather than cross-validation.
- No inference API, UI entry point, automated tests, or production deployment configuration is currently documented in this repository.

## 🗺️ Roadmap

Potential next steps:

- [ ] Build a simple Streamlit interface for entering customer queries.
- [ ] Expose inference through a FastAPI endpoint.
- [ ] Add confidence scores and a manual-review threshold.
- [ ] Compare TF-IDF + Logistic Regression with other text classifiers.
- [ ] Add cross-validation, a confusion matrix, and error analysis.
- [ ] Add unit tests and automated checks.
- [ ] Package the classifier and vectorizer in a single scikit-learn `Pipeline`.
- [ ] Explore answer suggestions or knowledge-base retrieval as a separate feature.

## 👨‍💻 Author

**Gurnoor Singh**  
AI/ML Enthusiast & Computer Science Student

GitHub: [@GSgurnoorsingh](https://github.com/GSgurnoorsingh)

---

<div align="center">

**Built to explore practical NLP, supervised learning, and customer support automation.**

</div>
