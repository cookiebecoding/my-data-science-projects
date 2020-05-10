# Decision Trees and Random Forest Project

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px

# Import data
loans = pd.read_csv('loan_data.csv')
# loans.head()
# loans.info()
# loans.describe()

# Exploratory Data Analysis
# Plotting some different charts + experimenting with different libraries to visually explore the data.

# Histogram with Plotly Graph Objects
fig_go = go.Figure()
fig_go.add_trace(
    go.Histogram(
        x = loans[loans['credit.policy'] == 1]['fico'],
        name ='credit.policy = 1',
        opacity = 0.8
    )
)
fig_go.add_trace(
    go.Histogram(
        x = loans[loans['credit.policy'] == 0]['fico'],
        name = 'credit.policy = 0',
        opacity = 0.8
    )
)
fig_go.update_layout(
    barmode='overlay',
    xaxis_title_text='fico',
    margin = dict(b=40, l=40, t=40, r=40),
    legend = dict(x=0.75, y=0.995, bgcolor = 'rgba(0,0,0,0)')
    )

# Same Histogram with Plotly Express
fig_px = px.histogram(loans, x='fico', color='credit.policy', barmode='overlay')
fig_px.show()

# Same Histogram with Seaborn
sns.distplot(loans[loans['credit.policy'] == 1]['fico'], kde=False, label='credit.policy = 1')
sns.distplot(loans[loans['credit.policy'] == 0]['fico'], kde=False, label='credit.policy = 0')
plt.legend()

# Same Histogram with Pyplot
loans[loans['credit.policy'] == 1]['fico'].hist(label='credit.policy = 1', bins=40)
loans[loans['credit.policy'] == 0]['fico'].hist(label='credit.policy = 0', bins=40)
plt.legend()
plt.xlabel('fico')

# Similar chart for the not.fully.paid variable
sns.distplot(loans[loans['not.fully.paid'] == 1]['fico'], kde=False, label='not.fully.paid = 1')
sns.distplot(loans[loans['not.fully.paid'] == 0]['fico'], kde=False, label='not.fully.paid = 0')
plt.legend()

# Countplot of loans by purpose for both not.fully.paid == 1 and not.fully.paid == 0
plt.figure(figsize=(12,5))
sns.countplot(data=loans, x=loans['purpose'], hue='not.fully.paid')

# Relation between FICO score and interest rate
sns.jointplot(x='fico', y='int.rate', data=loans)

# Relations between all variables
loans_pairplot = sns.pairplot(loans, hue='not.fully.paid')
# loans_pairplot.savefig("loans_pairplot.png")

# Transform categorical variables into dummy variables
cat_features = ['purpose']
df = pd.get_dummies(loans, columns=cat_features, drop_first=True)
df.info()

# Split data into training and test data
from sklearn.model_selection import train_test_split
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Predictions and Evaluation of Decision Tree
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# Train Random Forest model
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

# Predictions and Evaluation of Random Forest
preds_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, preds_rfc))
print(classification_report(y_test, preds_rfc))
