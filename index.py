import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

