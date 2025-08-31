# Task 5: Classification - Sentiment Analysis with Logistic Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

# Sample dataset
data = {
    "Review Text":[
        "I love this product, it is excellent!",
        "Worst purchase ever, very disappointed",
        "Good quality and fast shipping",
        "Terrible experience, not recommended",
        "I am very happy with this item",
        "Awful, waste of money",
        "Absolutely fantastic, exceeded expectations",
        "Not worth the price at all"
    ],
    "Sentiment":["positive","negative","positive","negative","positive","negative","positive","negative"]
}

df = pd.DataFrame(data)
df.to_csv("reviews.csv", index=False)

# Features and target
X = df["Review Text"]
y = df["Sentiment"]

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vec,y,test_size=0.25,random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Eval
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred, pos_label="positive")
rec = recall_score(y_test,y_pred, pos_label="positive")
f1 = f1_score(y_test,y_pred, pos_label="positive")

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print(classification_report(y_test,y_pred))

# Confusion matrix plot
ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap="Blues")
plt.title("Confusion Matrix - Sentiment Analysis")
plt.savefig("sentiment_confusion_matrix.png")
plt.close()
