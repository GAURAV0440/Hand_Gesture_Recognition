# Real-Time Hand Gesture Recognition

This project is a real-time Python application that detects static hand gestures using your webcam. It was built as part of the AI Intern Assessment.


## Developed By

**Name:** Gaurav  
**Submission for:** Bhatiyani AI Intern Assessment  
**Problem Statement:** Real-Time Static Hand Gesture Recognition  
**Demo Video:** [Watch the Demo](https://www.loom.com/share/167cca00757a4df8b60aaf423a5e3213?sid=9b80489c-c289-4e23-a01d-e682dfbd0e25)


## What It Does

- Uses your webcam to capture hand gestures in real-time
- Tracks 21 hand landmarks using MediaPipe
- Classifies 4 static gestures using a trained SVM model:
  - ✊ Fist
  - ✋ Open Palm
  - ✌ Peace Sign
  - 👍 Thumbs Up
- Displays the gesture name on the screen


## Technology Used

| Component        | Tool / Library      |
|------------------|---------------------|
| Hand Detection   | MediaPipe (Google)  |
| Video Input      | OpenCV              |
| Feature Handling | NumPy & Pandas      |
| Classifier       | SVM (scikit-learn)  |
| Model Saving     | Joblib              |


## Why These Tools?

- **MediaPipe** provides fast, accurate and memory-efficient hand tracking.
- **SVM (Support Vector Machine)** gives high accuracy on structured data like landmark points.
- All tools are lightweight and run easily on Ubuntu with no GPU or cloud dependencies.


## How to Run

### 1. Setup

git clone <your-repo-link>
cd hand_gesture_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


# 2. Collect Gesture Samples

Run "python3 collect_data.py"
Enter gesture label like "fist" or "palm"
Press 's' to save a sample, 'q' to quit

Repeat this for all 4 gestures.

# 3. Train the Model
Run "python3 train_model.py"

This creates:

gesture_model.pkl (SVM model)

label_encoder.pkl (label decoder)

# 4. Run Real-Time Gesture App
Run "python3 main.py"

## Project Structure

hand_gesture_project/
├── main.py
├── collect_data.py
├── train_model.py
├── gesture_model.pkl
├── label_encoder.pkl
├── data/
│   └── gesture_data.csv
├── requirements.txt
├── README.md
