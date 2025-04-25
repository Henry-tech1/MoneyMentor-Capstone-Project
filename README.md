## TITLE: MONEYMENTOR(A PERSONAL FINANCE ADVISOR MODEL)
### Problem Description:

Many people often struggle with their finances due to a combination of factors including:

- Lack of financial literacy and the emotional aspects that influence spending and investment decisions.- 
- Societal pressures and the difficulty of sticking to a budget, leading to living beyond one's means and accumulating insufficient savings.
- Poor debt management and the impact of unexpected life events.
- Underlying psychological factors and past experiences that shape financial behaviors and outcomes.

### Objective:
This model aims to empower individuals to manage their money more effectively by providing personalized, accessible, and automated financial guidance. Many people lack financial literacy or access to professional advice, which leads to poor budgeting, saving, or debt management. This model analyzes a user’s income, expenses, and financial habits to offer tailored recommendations, encouraging better financial habits and promoting long-term stability. By making financial planning more inclusive and affordable, such a model can help reduce debt, improve savings, and support overall financial well-being.

### Expected Outcome:
Build a financial system that automatically labels financial transactions, tracks spending habits over time, and provides actionable suggestions to help users manage their money more effectively.Incorporate automatic savings adjustments, updating how much to save based on real-time patterns, goals, or obligations.Provide real-time nudges, such as, "You're close to overspending this week."

### Data Loading and Exploration
The dataset, "Finances data.csv," was loaded into a pandas DataFrame. The dataset contains 20,000 entries and 27 columns. The columns include financial information such as income, age, expenses, and potential savings.
- Installed necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, silhouette_score
from sklearn.decomposition import PCA

Loaded data:
df = pd.read_csv('/Users/HM/Desktop/Finances data/Finances data.csv')

#### Data Exploration
df.shape
df.info()
df.head()
- The first few rows of the dataset show various financial attributes for each individual, including income, expenses across different categories, and potential savings.

### Data Preprocessing
- Categorical Data Encoding: The Occupation and City_Tier columns were identified as categorical variables. These columns were one-hot encoded to convert them into a numerical format suitable for machine learning models.

categorical_cols = ['Occupation', 'City_Tier']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

### Feature Scaling: 
To ensure that all features contribute equally to the model training process, the dataset was standardized using the StandardScaler. This was done by creating a pipeline.

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('Desired_Savings')
X = df[numerical_cols]
y = df['Desired_Savings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Methodology
The project involves several key tasks:

1. Spending Pattern Analysis:
Visualized spending distribution across different categories and identify potential areas for savings.Employed clustering techniques to segment users based on their spending behavior.
2. Savings Behavior Analysis:Predicted desired savings using regression models.Evaluated the accuracy of the predictions.
3. Personalized Financial Recommendations:Developed a system to provide tailored advice.Spending Pattern Analysis
4. Visualization of Spending Distribution: A plot was created to visualize how spending is distributed across different categories.

expense_categories = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']
total_spending = df[expense_categories].sum()
plt.figure(figsize=(10, 6))
total_spending.sort_values(ascending=True).plot(kind='barh')
plt.title('Total Spending per Category')
plt.xlabel('Total Spending')
plt.ylabel('Category')
plt.show()

- This visualization helps in understanding which spending categories are the highest and lowest.

5. Clustering for User Segmentation: K-means clustering was used to segment users based on their spending patterns. The optimal number of clusters was determined using the elbow method and silhouette scores.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_for_clustering)

# Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Silhouette Score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
The elbow method and silhouette score plots help in selecting the number of clusters.  In this case, 3 clusters were chosen.  The users are then segmented into these 3 clusters and the clusters added as a new column in the dataframe.Savings Behavior AnalysisPrediction of Desired Savings:  A RandomForestRegressor model was trained to predict the Desired_Savings.  The data was split into training and testing sets, and the model's performance was evaluated using Mean Absolute Error (MAE) and R-squared.numerical_cols = [
    'Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance',
    'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
    'Healthcare', 'Education', 'Miscellaneous', 'Desired_Savings_Percentage',
    'Disposable_Income', 'Potential_Savings_Groceries',
    'Potential_Savings_Transport', 'Potential_Savings_Eating_Out',
    'Potential_Savings_Entertainment', 'Potential_Savings_Utilities',
    'Potential_Savings_Healthcare', 'Potential_Savings_Education',
    'Potential_Savings_Miscellaneous'
]
X = df[numerical_cols]
y = df['Desired_Savings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, R²: {r2}")
The  RandomForestRegressor model was then optimized using grid search.param_grid = {
     'n_estimators': [100, 200],
     'max_depth': [None, 10, 20],
     'min_samples_split': [2, 5],
 }
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
print("Best parameters:", best_params)

best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, R²: {r2}")
Results

Spending Pattern Analysis: The bar chart illustrates the distribution of spending across different categories, showing that Rent and Groceries are the highest expenses.  The K-means clustering segmented users into 3 distinct groups based on their spending behavior.
Savings Behavior Analysis: The RandomForestRegressor model, with optimized parameters, achieved an MAE of 2797 and an R2 of 0.782, indicating a reasonably accurate prediction of desired savings.

### Conclusion
The Personal Finance Advisor Model provides valuable insights into users' financial behavior and offers personalized recommendations. The spending pattern analysis highlights areas where users can cut costs, while the savings behavior analysis enables accurate prediction of desired savings. The refined Random Forest Regressor model predicts desired savings with reasonable accuracy.  Further improvements, such as incorporating user feedback and additional data sources, could enhance the model's performance and usability.

### Future Directions
- Deploy the model via a web dashboard or mobile app

- Integrate banking APIs for real-time transaction monitoring

- Use reinforcement learning to dynamically optimize savings

- Add NLP components to process user queries or goals

### Contributions
Contributions of all kinds are welcome —feature ideas, bug fixes, or optimization strategies. Fork this repo or submit pull requests!

For collaboration or questions, please get in touch via GitHub or any preferred platform.
