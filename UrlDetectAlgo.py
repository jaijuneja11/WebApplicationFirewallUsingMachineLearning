from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse
import joblib


import matplotlib.pyplot as plt

def loadFile(name):
    directory = os.getcwd()
    filepath = os.path.join(directory, name)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result

def test_custom_queries(customQueries, vectorizer, lgs):
    # Vectorize custom queries
    X_custom = vectorizer.transform(customQueries)

    # Make predictions
    predicted_custom = lgs.predict(X_custom)

    # Print the predictions
    for i, query in enumerate(customQueries):
        if predicted_custom[i] == 1:
            print(f"Query '{query.strip()}' is predicted as MALICIOUS")
        else:
            print(f"Query '{query.strip()}' is predicted as BENIGN")

badQueries = loadFile('badQueries.txt')
validQueries = loadFile('goodQueries.txt')

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))
allQueries = badQueries + validQueries
yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
yGood = [0 for i in range(0, len(validQueries))]
y = yBad + yGood
queries = allQueries

vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
X = vectorizer.fit_transform(queries)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting data

badCount = len(badQueries)
validCount = len(validQueries)

lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0}) # class_weight='balanced')
lgs.fit(X_train, y_train) #training our model

##############
# Evaluation #
##############

predicted = lgs.predict(X_test)
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(lgs, "trained_model.joblib")


fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)




print("Bad samples: %d" % badCount)
print("Good samples: %d" % validCount)
print("Baseline Constant negative: %.6f" % (validCount / (validCount + badCount)))
print("------------")
print("Accuracy: %f" % lgs.score(X_test, y_test))  #checking the accuracy
print("Precision: %f" % metrics.precision_score(y_test, predicted))
print("Recall: %f" % metrics.recall_score(y_test, predicted))
print("F1-Score: %f" % metrics.f1_score(y_test, predicted))
print("AUC: %f" % auc)
print("------------")


def loadFile(name):
    directory = os.getcwd()
    filepath = os.path.join(directory, name)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   # Converting URL encoded data to simple string
        result.append(d)
    return result

def test_custom_queries(customQueries, vectorizer, lgs):
    # Vectorize custom queries
    X_custom = vectorizer.transform(customQueries)

    # Make predictions
    predicted_custom = lgs.predict(X_custom)

    # Print the predictions
    for i, query in enumerate(customQueries):
        if predicted_custom[i] == 1:
            print(f"Query '{query.strip()}' is predicted as MALICIOUS")
        else:
            print(f"Query '{query.strip()}' is predicted as BENIGN")

# Load custom queries
customQueries = loadFile('customQueries.txt')  # Adjust filename as per your custom queries file

# Test custom queries
test_custom_queries(customQueries, vectorizer, lgs)
