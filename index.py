import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
main_df = pd.read_csv('TMDB_movie_dataset_v11.csv', sep=',', on_bad_lines='warn')
df = main_df[main_df['vote_average'] != 0]

df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

df = df.drop(['id', 'vote_count', 'status', 'release_date', 'revenue', 'backdrop_path', 'budget', 
              'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path', 'tagline', 
              'production_companies', 'production_countries', 'spoken_languages', 'keywords'], axis=1)

df['genres'] = df['genres'].fillna('unknown')
label_encoder = LabelEncoder()
df['genres'] = label_encoder.fit_transform(df['genres'])
df['original_language'] = label_encoder.fit_transform(df['original_language'])

X = df.drop(['title', 'vote_average'], axis=1)
y = df['vote_average'].apply(lambda x: 1 if x >= 7 else 0) 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

# Evaluation metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

print("Random Forest Evaluation Metrics")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print("\nClassification Report:\n", rf_report)

# Evaluation metrics for Decision Tree
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_precision = precision_score(y_test, dt_y_pred)
dt_recall = recall_score(y_test, dt_y_pred)
dt_f1 = f1_score(y_test, dt_y_pred)
dt_report = classification_report(y_test, dt_y_pred)

print("\nDecision Tree Evaluation Metrics")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print("\nClassification Report:\n", dt_report)

# Confusion matrix for Random Forest
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Confusion matrix for Decision Tree
dt_conf_matrix = confusion_matrix(y_test, dt_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(dt_conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
