

# Spam Detection Model

## Introduction
Welcome to the Spam Detection Model project! This repository provides a robust machine learning solution for classifying email text as "Spam" or "Not Spam (Ham)" using advanced natural language processing (NLP) techniques and a Logistic Regression classifier. The project includes a user-friendly Streamlit web application for real-time predictions.

## Key Features
- **Intuitive Web Interface**: Easily interact with the model using a modern Streamlit app.
- **Comprehensive Text Preprocessing**: Utilizes NLTK for tokenization, stopword removal, and lemmatization to ensure high-quality input for the model.
- **Accurate Machine Learning Model**: Logistic Regression trained on vectorized email data for reliable spam detection.
- **Instant Results**: Get immediate feedback on your email text classification.

## Getting Started
Follow these steps to set up and use the Spam Detection Model:

### Prerequisites
- Python 3.7 or higher
- Required Python packages (see `requirement.txt`)

### Installation
1. **Clone the repository** or download the source code.
2. **Install dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

### Running the Application
1. Ensure `spam_detection_model.pkl` and `vectorizer.pkl` are present in the project directory.
2. Launch the Streamlit app:
    ```bash
    streamlit run spam_detection.py
    ```
3. Enter your email text in the provided text area and click **Predict** to view the classification result.

## Example Usage
```
Subject: Congratulations! You have won a prize.
Click here to claim your reward now.
```
Prediction: **SPAM**

## Project Structure
- `spam_detection.py`: Streamlit application script.
- `spam_detection_model.pkl`: Pre-trained machine learning model.
- `vectorizer.pkl`: Text vectorizer for feature extraction.
- `requirement.txt`: List of required Python packages.
- `Spam_detection_model/README.md`: Project documentation.
- `Spam_detection_model/LICENSE`: License information.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) for details.

## Acknowledgements
- Developed by LovishTech
- Built with Streamlit, scikit-learn, NLTK, pandas, and joblib
