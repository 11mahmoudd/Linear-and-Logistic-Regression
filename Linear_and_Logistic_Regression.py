#%%

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import SGDClassifier
#%%

#%%
data = pd.read_csv('co2_emissions_data.csv')

print(data.shape)
cleaned = data.dropna()
cleaned= cleaned.drop_duplicates()
print(cleaned.shape)
#%%
cleaned.drop(columns=['Fuel Consumption Comb (mpg)'], inplace=True)
cleaned
#%%
numerical=['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
           'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)','CO2 Emissions(g/km)']

numericalFeatures=['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                  'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)']

nonNumerical=['Make','Model','Vehicle Class','Transmission','Fuel Type','Emission Class']

#%%
print(cleaned[numericalFeatures].describe())
#%%
cleaned[numericalFeatures].boxplot(figsize=(15, 6))
mp.title("Boxplot of Numerical Features")
mp.ylabel("Values")
mp.show()
#%%
sns.pairplot(cleaned,diag_kind='hist',hue='Emission Class')
mp.show()

#%%
sns.heatmap(cleaned[numerical].corr(), annot=True,cmap='summer')
mp.show()
#%%
from sklearn.preprocessing import LabelEncoder

toEncoded=['Transmission','Fuel Type','Emission Class']

for col in toEncoded:
    cleaned[col] = LabelEncoder().fit_transform(cleaned[col])
# cleaned
# result = cleaned[cleaned['Emission Class']==3]
# print(result)
#%%
feature=['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                  'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)']+toEncoded[0:2]
target=['CO2 Emissions(g/km)']+toEncoded[2:]
numericalTarget=['CO2 Emissions(g/km)']

nonNumericalFeatures=['Transmission','Fuel Type']

x=cleaned[feature]
y=cleaned[target]

x, y = shuffle(x, y, random_state=42)

# print(target[1])
# x
# y
#%%
input=['Engine Size(L)','Fuel Consumption Comb (L/100 km)']
output=['CO2 Emissions(g/km)']

#%%

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_num=x_train[numericalFeatures]
x_train_str=x_train[nonNumericalFeatures]
x_test_num=x_test[numericalFeatures]
x_test_str=x_test[nonNumericalFeatures]
print(x_train_num)
print(x_train_str)
y_train_class=y_train[target[1]]
y_test_class=y_test[target[1]]
y_test=y_test[numericalTarget]
y_train=y_train[numericalTarget]

# print(max(y_test_class))
#%%
scaler_x = MinMaxScaler()

scaled_x_train = scaler_x.fit_transform(x_train_num)

scaled_x_test = scaler_x.transform(x_test_num)


x_train_scaled_df=pd.DataFrame(scaled_x_train)
x_test_scaled_df=pd.DataFrame(scaled_x_test)

x_train_input=x_train_scaled_df.iloc[:, [0, 4]]
x_test_input=x_test_scaled_df.iloc[:, [0, 4]]

print(x_train_input)

print(x_test_input)

print(x_train_scaled_df)


print(y_train.to_numpy().shape)
# print(x_train)
# print(y_test)
print(y_test_class.shape)
print(y_train_class.shape)

#%% md
# 
#%%

#%%
x_train_df = pd.DataFrame(scaled_x_train, columns=numericalFeatures)
x_train_df.boxplot(figsize=(15, 6))
mp.title("Boxplot of Numerical Features")
mp.ylabel("Values")
mp.show()
#%%

#%%
class LinearRegression():
    def __init__(self, alpha=0.01, max_iterations=1000):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.theta = None
        self.cost_history = []

    def hypothesis(self, x):
        hyp= (np.dot(x,self.theta))
        return hyp
        
    def cost_function(self,x,hyp,y):
        cost = (1/len(x)*(np.sum(np.pow(hyp - y, 2))))
        return cost
    
    def gradient(self,x,y):
         m, n = x.shape
         self.theta = np.zeros(n)
         for i in range(self.max_iterations):
             h=self.hypothesis(x)
             for j in range(len(self.theta)):
                 part= 1/m * sum( (h-y) * x[:,j] )
                 self.theta[j] = self.theta[j] - self.alpha *part 
             cost=self.cost_function(x,h,y)
             self.cost_history.append(cost)
         return self.theta
    
    def fit(self, x, y):
        x = np.c_[np.ones((x.shape[0], 1)), x] 
        y=y.to_numpy()
        y = y.flatten() 
        self.theta=self.gradient(x,y)
    
    def predict(self, x):   
         x = np.c_[np.ones((x.shape[0], 1)), x]
         return self.hypothesis(x)
    
    
    
#%%


print(x_train_input.shape)
print(x_test_input.shape)
print(y_train.shape)
print(y_test.shape)

max_iterations=10000
learning_rate=0.001

model = LinearRegression(max_iterations=max_iterations,alpha=learning_rate)

model.fit(x_train_input, y_train)

predictions = model.predict(x_test_input)  
print('--------------') 
print(f'predictions: {predictions}')
print('--------------')

# test_predictions = model.predict(x_test_input)

r2 = r2_score(y_test, predictions)

print(f"R² score: {r2}")

#%%

#%%
print(model.theta)
#%%
mp.figure(figsize=(10, 6))
mp.plot(model.cost_history, color="blue")
mp.xlabel("Iterations")
mp.ylabel("Cost")
mp.title("Cost Function Over Iterations")
mp.grid(True)
mp.show()
#%%

model = SGDClassifier(max_iter=1000 ,alpha=0.001,random_state=42)

model.fit(x_train_input, y_train_class)
y_pred = model.predict(x_test_input)

accuracy = accuracy_score(y_test_class, y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy * 100:.2f}%")