import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle 
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("insurance.csv")

df.head()

df['charge_Per_age']=df['charges']/df['age']

#df['charge_Per_bmi']=df['charges']/df['bmi']
#df =df.drop(["bmi"],axis=1)
df.columns

df.head()

#statsmodel 
import statsmodels.formula.api as smf

model=smf.ols("charges ~ age+sex+children+smoker+region+charge_Per_age",data=df).fit()

model.params

#df['charges'] = df['charges'].astype(int)

df.head()

le= preprocessing.LabelEncoder()

df["sex"]=le.fit_transform(df["sex"])

df["smoker"]=le.fit_transform(df["smoker"])

df["region"]=le.fit_transform(df["region"])

df.head()

# Select independent and dependent variable
y=df[["charges","charge_Per_age"]]
x=df.drop(["sex","charges","charge_Per_age"],axis=1)

# Split the dataset into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# Instantiate the model
classifier = RandomForestRegressor()

# Fit the model
classifier.fit(x_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

df.to_csv("insurance.csv")