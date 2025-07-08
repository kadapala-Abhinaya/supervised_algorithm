import pandas as pd
from sklearn.tree import DecisionTreeRegressor
data={
    'sqrfeet':[1400,1200,1600,1800,1100],
    'rooms':[3,2,4,5,1],
    'age':[10,5,15,7,20],
    'price':[75,65,80,90,60]
}
df=pd.DataFrame(data)
x=df[['sqrfeet','rooms','age']]
y=df['price']
model = DecisionTreeRegressor()
model.fit(x, y)
new_house=[[1600,5,20]]
prediction_price=model.predict(new_house)
print(prediction_price)