import joblib

try:
    # Load the files
    model = joblib.load('naive_bayes_model/naive_bayes_model.pkl')
    vectorizer = joblib.load('naive_bayes_model/tfidf_vectorizer.pkl')

    test_text = ["breaking news about the stock market and corporate profits"]
    test_vec = vectorizer.transform(test_text)
    prediction = model.predict(test_vec)
    print(prediction[0])

except Exception as e:
    print("Error occured while loading Naive Bayes model")