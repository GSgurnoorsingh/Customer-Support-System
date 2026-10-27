import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/dataset.csv"

if not os.path.exists(DATA_PATH):
    print("ERROR: dataset.csv not found!")
    print("Please put your dataset inside the data folder.")
    exit()

df = pd.read_csv(DATA_PATH)

print("\n==============================")
print("       SUPPORTIQ TRAINING")
print("==============================")

print("\nDataset shape:", df.shape)

# Remove missing values
df = df.dropna(subset=["query", "category"])

# Convert everything to string
df["query"] = df["query"].astype(str)
df["category"] = df["category"].astype(str)

print("Number of training examples:", len(df))
print("Number of categories:", df["category"].nunique())

print("\nCategories:")
for category in sorted(df["category"].unique()):
    print("-", category)


# --------------------------------------------------
# 2. Split dataset
# --------------------------------------------------

X = df["query"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. TF-IDF Vectorizer
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# --------------------------------------------------
# 4. Train Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining model...")

model.fit(X_train_vectorized, y_train)


# --------------------------------------------------
# 5. Evaluate
# --------------------------------------------------

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("       MODEL RESULTS")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# --------------------------------------------------
# 6. Save model
# --------------------------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

labels = sorted(df["category"].unique())

joblib.dump(labels, "model/labels.pkl")


print("\n==============================")
print("       TRAINING COMPLETE")
print("==============================")

print("\nSaved:")
print("model/model.pkl")
print("model/vectorizer.pkl")
print("model/labels.pkl")