from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model_path = "../url injection detection using ML/trained_model.joblib"
vectorizer_path = "../url injection detection using ML/tfidf_vectorizer.joblib"
lgs = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)



def classify_url(url):
    # Make sure the TF-IDF vectorizer is fitted
    if not hasattr(vectorizer, 'vocabulary_'):
        raise ValueError("TF-IDF vectorizer is not fitted")

    # Check if the URL is not empty
    if url is None or url.strip() == "":
        raise ValueError("URL is empty")

    # Vectorize the URL
    X_url = vectorizer.transform([url])

    # Make prediction
    prediction = lgs.predict(X_url)[0]

    # Return the classification result
    return prediction



@app.route('/', methods=["POST", "GET"])
def classify_url_route():
    # Get the URL from the request parameters
    url = request.args.get("url")

    # Perform classification
    classification = classify_url(url)

    # Return the classification result
    if classification == 1:
        return jsonify({"classification": "MALICIOUS"})
    else:
        return jsonify({"classification": "BENIGN"})

if __name__ == "__main__":
    app.run(debug=True)
