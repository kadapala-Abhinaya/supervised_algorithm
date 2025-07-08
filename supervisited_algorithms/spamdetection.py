import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

data = {
    'message': [
        "Win a free lottery ticket", "hi, find your attachment file", "kudos, you won a car",
        "meet you at 10am tomorrow", "claim your free gift", "terrorist planned bomb blast in mrecw",
        "abhinaya is very good girl"
    ],
    'status': ['spam', 'not spam', 'spam', 'not spam', 'spam', 'spam', 'not spam']
}

df = pd.DataFrame(data)
df['status'] = df['status'].map({'spam': 1, 'not spam': 0})
x = df['message']
y = df['status']
vectorizer = CountVectorizer()
x_vectorizer = vectorizer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_vectorizer, y, test_size=0.3, random_state=4)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("accuracy", accuracy_score(y_test, y_pred))
sample_message=["abhinaya is very good girl"]
sample_vector=vectorizer.transform(sample_message)
prediction=model.predict(sample_vector)
print(prediction[0])

