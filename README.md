# Spam Detection Model

## Introduction

Welcome to the Spam Detection Model project! This repository provides a robust machine learning solution for classifying email text as **"Spam"** or **"Not Spam (Ham)"** using advanced natural language processing (NLP) techniques and a Logistic Regression classifier. The project includes a user-friendly **Streamlit web application** for real-time predictions and a **Jupyter Notebook** for experimentation and model exploration.

## Key Features

-   **Intuitive Web Interface**: Easily interact with the model using a modern Streamlit app.
    
-   **Comprehensive Text Preprocessing**: Utilizes NLTK for tokenization, stopword removal, and lemmatization to ensure high-quality input for the model.
    
-   **Accurate Machine Learning Model**: Logistic Regression trained on vectorized email data for reliable spam detection.
    
-   **Jupyter Notebook Support**: `spam_detection.ipynb` allows you to explore data, experiment with preprocessing, train models, and visualize results.
    
-   **Instant Results**: Get immediate feedback on your email text classification.
    

## Getting Started

### Prerequisites

-   Python 3.7 or higher
    
-   Required Python packages (see `requirement.txt`)
    

### Installation

1.  **Clone the repository** or download the source code:
    
    `git clone <repository_url>`
    
2.  **Install dependencies**:
    
    `pip install -r requirement.txt`
    

### Running the Streamlit Application

1.  Ensure `spam_detection_model.pkl` and `vectorizer.pkl` are present in the project directory.
    
2.  Launch the Streamlit app:
    
    `streamlit run spam_detection.py`
    
3.  Enter your email text in the provided text area and click **Predict** to view the classification result.
    

### Using the Jupyter Notebook

1.  Open `spam_detection.ipynb` in Jupyter Lab or Jupyter Notebook.
    
2.  Explore data preprocessing, model training, evaluation, and predictions step by step.
    
3.  Modify parameters or try different ML algorithms directly in the notebook.
    

## How to Train Your Own Model

If you want to train the spam detection model from scratch or update it with new data, follow these steps:

1.  Open `spam_detection.ipynb`.
    
2.  Load your dataset containing emails and their labels (`Spam` or `Ham`).
    
3.  Preprocess the text:
    
    -   Tokenization
        
    -   Stopword removal
        
    -   Lemmatization
        
4.  Vectorize the text using `CountVectorizer` or `TfidfVectorizer`.
    
5.  Train a machine learning model, e.g., Logistic Regression, on the vectorized features.
    
6.  Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
    
7.  Save the trained model and vectorizer using `joblib`:
    
    `import joblib  joblib.dump(model, 'spam_detection_model.pkl') joblib.dump(vectorizer, 'vectorizer.pkl')`
    
8.  Use the saved files in the Streamlit app for real-time predictions.
    

## Example Usage

`Subject: Congratulations! You have won a prize. Click here to claim your reward now.`

**Prediction:** SPAM

## Project Structure

-   `spam_detection.py` : Streamlit application script
    
-   `spam_detection.ipynb` : Jupyter Notebook for experimentation and model development
    
-   `spam_detection_model.pkl` : Pre-trained machine learning model
    
-   `vectorizer.pkl` : Text vectorizer for feature extraction
    
-   `requirement.txt` : List of required Python packages
    
-   `Spam_detection_model/README.md` : Project documentation
    
-   `Spam_detection_model/LICENSE` : License information
    

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) for details.

## Acknowledgements

-   Developed by **LovishTech**
    
-   Built with **Streamlit**, **scikit-learn**, **NLTK**, **pandas**, **joblib**, and **Jupyter Notebook**
