

import pickle
# Import the required packages
import pandas as pd # for handling data in the form of tables
import numpy as np # For handling matrix
from sklearn.model_selection import train_test_split # For spliting the data into train and test
from sklearn.linear_model import LinearRegression # for using the model
from sklearn import metrics
import matplotlib.pyplot as plt # for plotting
import seaborn as sns

from sklearn.metrics import accuracy_score
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Upload the data 
dataset=pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",")
dataset.head()
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
dataset.dropna(axis=0, how= 'all', inplace=True)
dataset.dropna(axis=1, inplace=True)
dataset.replace(to_replace= -200, value= np.NaN, inplace= True)
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

dataset.drop(['Time'], axis=1, inplace=True)
dataset.drop(['NMHC(GT)'], axis= 1, inplace= True)

dataset.fillna(method='ffill', inplace= True)
dataset.isnull().any().any()
dataset.isna().any()
dataset=dataset.fillna(method="ffill")
sns.distplot(dataset["CO(GT)"])
# Select the independendent variables
X = dataset[['CO(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T','RH', 'AH']].values
x = pd.DataFrame(X)
# Select the depended variables
Y = dataset['C6H6(GT)'].values
y = pd.DataFrame(Y)
# divide the data into train and test
 
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=0)



#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Fitting model with trainig data
regressor.fit(X_train, Y_train)
p=regressor.predict(X_test)
print(r2_score(Y_test,p))
print('Score: ', regressor.score(X_test, Y_test))
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6,1360,1046,116,1056,113,1692,1268,13.6,48.9,0.7578]]))


from sklearn.tree import DecisionTreeRegressor
dregressor = DecisionTreeRegressor(random_state=0)
dregressor.fit(X_train,Y_train)
pv=dregressor.predict(X_test)
print(r2_score(Y_test,pv))
pickle.dump(dregressor, open('model2.pkl','wb'))


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(max_depth=2, random_state=0)
clf.fit(X_train, Y_train)
y_pred=clf.predict(X_test)
print(r2_score(Y_test,y_pred))
pickle.dump(clf, open('model3.pkl','wb'))

# Loading model to compare the results
modelr = pickle.load(open('model3.pkl','rb'))
print(modelr.predict([[2.6,1360,1046,116,1056,113,1692,1268,13.6,48.9,0.7578]]))



from sklearn.ensemble import GradientBoostingRegressor
clf=GradientBoostingRegressor()
clf.fit(X_train, Y_train)
y_pred=clf.predict(X_test)
print(r2_score(Y_test,y_pred))
pickle.dump(clf, open('model4.pkl','wb'))

# Loading model to compare the results
modeld = pickle.load(open('model4.pkl','rb'))
print(modeld.predict([[2.6,1360,1046,116,1056,113,1692,1268,13.6,48.9,0.7578]]))




# Loading model to compare the results
modeld = pickle.load(open('model4.pkl','rb'))
print(modeld.predict([[2.6,1360,1046,116,1056,113,1692,1268,13.6,48.9,0.7578]]))