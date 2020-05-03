"""
Logistic Regression Project

Simple exercise on Logistic Regression in which we will predict whether or not a particular user clicked on a website ad.
The project uses a fake dataset, which can be found in the project folder.

The advertising dataset contains the following features:
* 'Daily Time Spent on Site': consumer time on site in minutes
* 'Age': cutomer age in years
* 'Area Income': Avg. Income of geographical area of consumer
* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
* 'Ad Topic Line': Headline of the advertisement
* 'City': City of consumer
* 'Male': Whether or not consumer was male
* 'Country': Country of consumer
* 'Timestamp': Time at which consumer clicked on Ad or closed window
* 'Clicked on Ad': 0 or 1 indicated clicking on Ad
"""

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Import dataset
ad_data = pd.read_csv('advertising.csv')

# Check dataframe
# ad_data.head()
# ad_data.info()
# ad_data.describe()

# Explore data
# sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')

# Create model
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train, y_train)

# Execute predictions
predictions = logmodel.predict(X_test)

# Evaluate model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
