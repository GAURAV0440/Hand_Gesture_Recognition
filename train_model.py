import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("data/gesture_data.csv")

# Split into features and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Encode string labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# Save model & encoder
joblib.dump(clf, "gesture_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Detailed evaluation
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
