import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("student_data.csv")

# -----------------------------
# 1. Distribution of Final Grade
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=11, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.show()


# -----------------------------
# 2. Study Time vs Final Grade
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.show()


# -----------------------------
# 3. Failures vs Final Grade
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="failures", y="G3", data=df)
plt.title("Previous Failures vs Final Grade")
plt.xlabel("Number of Failures")
plt.ylabel("Final Grade")
plt.show()


# -----------------------------
# 4. Absences vs Final Grade
# -----------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x="absences", y="G3", data=df)
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.show()


# -----------------------------
# 5. Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show