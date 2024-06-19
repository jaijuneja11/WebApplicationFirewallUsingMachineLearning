# WebApplicationFirewallUsingMachineLearning

## Overview
This Python script demonstrates the implementation of a machine learning model for detecting malicious queries in web traffic. The model is trained using logistic regression and utilizes the TF-IDF vectorization technique for feature extraction from query strings.

## Requirements
- Python 3.x
- scikit-learn (sklearn)
- matplotlib

## Usage
1. Clone or download the repository.
2. Ensure you have the required dependencies installed (`pip install -r requirements.txt`).
3. Place your dataset files `badQueries.txt` and `goodQueries.txt` in the same directory as the script.
4. Run the script `malicious_query_detection.py`.

## Functionality
- The script loads the dataset files containing malicious and benign query strings.
- It preprocesses the data by converting URL-encoded strings to simple text.
- The data is split into training and testing sets using a specified ratio.
- TF-IDF vectorization is applied to convert the text data into numerical feature vectors.
- Logistic regression model is trained using the training data.
- Model evaluation metrics such as accuracy, precision, recall, F1-score, and AUC are computed.
- Optionally, you can uncomment the code to test custom queries.

## Output
The script prints various evaluation metrics of the trained model, including accuracy, precision, recall, F1-score, and AUC. Additionally, it plots the Receiver Operating Characteristic (ROC) curve.

---

Feel free to adjust or expand upon this README file to better suit your project's needs!
