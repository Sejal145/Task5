# Task 5: Classification - Student Pass/Fail Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Sample dataset
data = {
    "Study Hours":[5,10,2,8,7,3,12,15,1,9],
    "Attendance":[80,90,60,85,75,50,95,98,40,82],
    "Pass":[1,1,0,1,1,0,1,1,0,1]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("student_data.csv", index=False)

# Explore
print(df.head())

# Visualization
sns.scatterplot(data=df, x="Study Hours", y="Attendance", hue="Pass", style="Pass")
plt.title("Study Hours vs Attendance by Pass/Fail")
plt.savefig("student_scatter.png")
plt.close()

# Split
X = df[["Study Hours","Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Eval
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail","Pass"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Student Prediction")
plt.savefig("student_confusion_matrix.png")
plt.close()
