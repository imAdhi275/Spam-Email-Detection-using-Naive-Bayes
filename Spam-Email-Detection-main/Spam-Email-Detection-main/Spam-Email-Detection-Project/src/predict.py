def predict_email(text, preprocess_func, vectorizer, model, model_type="multinomial"):
    cleaned = preprocess_func(text)
    vec = vectorizer.transform([cleaned])

    if model_type == "gaussian":
        pred = model.predict(vec.toarray())[0]
        proba = model.predict_proba(vec.toarray())[0][1]
    else:
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]

    return ("SPAM" if pred == 1 else "HAM"), round(float(proba), 3)