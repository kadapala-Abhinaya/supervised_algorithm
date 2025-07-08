import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = {
    'maths': [78, 40, 50, 67, 23, 56, 70, 99, 54, 76],
    'physics': [80, 33, 57, 67, 90, 99, 23, 56, 79, 88],  # fixed column name
    'chemistry': [30, 45, 67, 78, 90, 99, 75, 43, 21, 77],
    'Result': ['fail', 'fail', 'pass', 'pass', 'fail', 'pass', 'fail', 'pass', 'fail', 'pass']
}
df = pd.DataFrame(data)
df['Result'] = df['Result'].map({'pass': 1, 'fail': 0})
x = df[['maths', 'physics', 'chemistry']]
y = df['Result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
new_stu=pd.DataFrame([[60,15,85]],columns=['maths','physics','chemistry'])
prediction=model.predict(new_stu)
print(prediction[0])