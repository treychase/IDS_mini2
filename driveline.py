# ### Step 0: Load Data and Libraries

# Install required libraries if not already installed
#pip install pandas matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import io

data = pd.read_csv('hp_obp.csv')

# ### Step 1: Exploratory Data Analysis (EDA)

# First 5 Rows of Data
print("#### First 5 Rows of Data")
print(data.head())

# Data Info
print("#### Data Info")
data_info = io.StringIO()
data.info(buf=data_info)
print(data_info.getvalue())

# Data Description
print("#### Data Description")
print(data.describe())

# Missing Values per Column
print("#### Missing Values per Column")
print(data.isnull().sum().to_frame("Null Count").T)

# Total Duplicate Rows
print("#### Total Duplicate Rows")
print(f"**{data.duplicated().sum()}**")

# ### Step 2: Data Preprocessing

# Filtering for pitching and peak biomechanics metrics, as well as primary keys for the athletes
cols = [col for col in data.columns if 'peak' in col.lower()]
cols += ['pitch_speed_mph', 'test_date', 'athlete_uid']

# Filter DataFrame for these columns (ignore missing columns)
filtered_cols = [col for col in cols if col in data.columns]
filtered_data = data[filtered_cols]

# Remove outliers in pitch_speed_mph using the IQR method
Q1 = filtered_data['pitch_speed_mph'].quantile(0.25)
Q3 = filtered_data['pitch_speed_mph'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_data = filtered_data[
    (filtered_data['pitch_speed_mph'] >= lower_bound) &
    (filtered_data['pitch_speed_mph'] <= upper_bound)
]

# Fill all nulls with mean value
filtered_data = filtered_data.fillna(filtered_data.mean(numeric_only=True))

print(filtered_data.head())
print(filtered_data.shape)
print(filtered_data.describe())

plt.figure(figsize=(8, 6))
plt.hist(filtered_data['pitch_speed_mph'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Pitch Speed (mph)")
plt.ylabel("Frequency")
plt.title("Histogram of Pitch Speed (mph)")

# Add a dashed red line for the mean
mean_speed = filtered_data['pitch_speed_mph'].mean()
plt.axvline(mean_speed, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_speed:.2f}")
plt.legend()
plt.show()

# ### Step 3: Implement Machine Learning

# #### Model 1: Classic Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Select only float columns except 'pitch_speed_mph'
float_cols = filtered_data.select_dtypes(include='float').columns.tolist()
if 'pitch_speed_mph' in float_cols:
    float_cols.remove('pitch_speed_mph')

X = filtered_data[float_cols]
y = filtered_data['pitch_speed_mph']

# Split data into train and test sets FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Pitch Speed (mph)")
plt.ylabel("Predicted Pitch Speed (mph)")
plt.title("Actual vs. Predicted Pitch Speed")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 1:1 line
plt.show()

# #### Model 2: Bayesian Regression

from sklearn.linear_model import BayesianRidge

# Train a Bayesian Ridge Regression model
bayes_model = BayesianRidge()
bayes_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_bayes = bayes_model.predict(X_test_scaled)

# Evaluate the Bayesian Ridge model
mse_bayes = mean_squared_error(y_test, y_pred_bayes)
r2_bayes = r2_score(y_test, y_pred_bayes)

print("Bayesian Ridge Mean Squared Error:", mse_bayes)
print("Bayesian Ridge R^2 Score:", r2_bayes)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_bayes, alpha=0.7, color='purple')
plt.xlabel("Actual Pitch Speed (mph)")
plt.ylabel("Predicted Pitch Speed (mph)")
plt.title("Bayesian Ridge: Actual vs. Predicted Pitch Speed")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='1:1 line')
plt.show()

# #### Model 3: Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Mean Squared Error:", mse_rf)
print("Random Forest R^2 Score:", r2_rf)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
plt.xlabel("Actual Pitch Speed (mph)")
plt.ylabel("Predicted Pitch Speed (mph)")
plt.title("Random Forest: Actual vs. Predicted Pitch Speed")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='1:1 line')
plt.legend()
plt.show()

# ### Model Comparison

results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Bayesian Ridge', 'Random Forest'],
    'Mean Squared Error': [mse, mse_bayes, mse_rf],
    'R^2 Score': [r2, r2_bayes, r2_rf]
})

print(results_df)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Linear Regression', color='blue', marker='o')        # Circle
plt.scatter(y_test, y_pred_bayes, alpha=0.6, label='Bayesian Ridge', color='purple', marker='^')   # Triangle
plt.scatter(y_test, y_pred_rf, alpha=0.6, label='Random Forest', color='green', marker='s')        # Square
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='1:1 Line')
plt.xlabel("Actual Pitch Speed (mph)")
plt.ylabel("Predicted Pitch Speed (mph)")
plt.title("Actual vs. Predicted Pitch Speed: Model Comparison")
plt.legend()
plt.show()

# ### Summary:
# Based on the results dataframe, all three models—Linear Regression, Bayesian Ridge, and Random Forest—demonstrate similar performance in predicting pitch velocity. The Random Forest model achieves the lowest mean squared error (27.07) and the highest R² score (0.69), indicating it explains slightly more variance and makes more accurate predictions than the linear models. Both Linear Regression and Bayesian Ridge perform nearly identically, with mean squared errors around 29.1 and R² scores around 0.67. Overall, while all models perform reasonably well, the Random Forest regressor provides the best predictive accuracy for pitch velocity in this dataset. However, since all models perform similarly, I would recommend the simplest model for cost efficiency, which would be classical linear