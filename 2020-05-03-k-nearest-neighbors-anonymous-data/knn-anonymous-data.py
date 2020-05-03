# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objs as go

# Import data
df = pd.read_csv('/Users/lizecoekaerts/Documents/Courses/00-Python/2019-07_Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data.csv')
#df.head()
#df.describe()
#df.info()

# Data exploration
#sns.pairplot(df, hue='TARGET CLASS', palette='bwr')

# Standardize the Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
#df_scaled.head()

# Split data into a training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['TARGET CLASS'], test_size=0.30)

# Create model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# Fit model to data
knn.fit(X_train, y_train)

# Predictions
predictions = knn.predict(X_test)

# Evaluations
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Choose K value
error_rate = []
for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predictions_i = knn.predict(X_test)
    error_rate.append(np.mean(predictions_i != y_test))

# Plot K values and corresponding error rates
fig = go.Figure(
    go.Scatter(
        y = error_rate,
        mode='lines+markers',
        line=dict(color='blue', dash='dot'),
        marker=dict(symbol='circle', color='red', line=dict(color='blue', width=1), size=10)),
    go.Layout(
        font=dict(size=10),
        title='Error Rate vs. K Value',
        title_yanchor='bottom',
        title_x=0.5,
        template='simple_white',
        xaxis=dict(title='K', mirror=True),
        yaxis=dict(title='Error Rate', mirror=True),
        width=700,
        height=400,
        margin=dict(b=20, l=20, r=20, t=40)))
fig.show()

# Retrain model with new K Value
knn = KNeighborsClassifier(n_neighbors=32)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

# Evaluate new model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
