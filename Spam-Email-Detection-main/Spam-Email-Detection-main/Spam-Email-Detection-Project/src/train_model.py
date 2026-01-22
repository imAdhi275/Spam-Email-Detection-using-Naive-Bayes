from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB

def get_vectorizers(max_features=5000):
    bow = CountVectorizer(max_features=max_features)
    tfidf = TfidfVectorizer(max_features=max_features)
    return bow, tfidf

def vectorize(vectorizer, X_train, X_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec

def train_multinomial_nb(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model

def train_gaussian_nb(X_train_vec, y_train):
    model = GaussianNB()
    model.fit(X_train_vec.toarray(), y_train)
    return model