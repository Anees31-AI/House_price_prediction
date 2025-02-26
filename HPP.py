import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Load dataset
df = pd.read_csv("Housing.csv")

# Display basic info
print(df.info())

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# Convert binary categorical columns ('yes' → 1, 'no' → 0)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})

# Apply One-Hot Encoding only once for all categorical columns
df = pd.get_dummies(df, drop_first=True)

# Print columns to verify
print(df.columns)
print(df.head())


#Now that the dataset is cleaned and preprocessed, we need to split it into training and testing sets.

#We will use train_test_split from sklearn.model_selection:

# Define features (X) and target variable (y)
X=df.drop("price",axis=1)
y=df["price"] #Target values

#Split into training (80%) and testing (20%) sets

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

#Display shapes to verify split

print("Training Features shape:",X_train.shape)
print("Testing Feature shape:",X_test.shape)
print("Training Target shape:",y_train.shape)
print("Testing Target shape:",y_test.shape)

#Training a Linear Regression Model

#Now that the data is split, let's train a Linear Regression model, which is a good starting point for house price prediction.

#Initialize the model
model=LinearRegression()

#Train the model
model.fit(X_train,y_train)

#Make Predictions
y_pred=model.predict(X_test)

#Evaluate the model
mae=mean_absolute_error(y_test,y_pred) #Mean Absolute error
mse=mean_squared_error(y_test,y_pred) #Mean squared error
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # ✅ Take square root manually
r2=r2_score(y_test,y_pred) #R-squared score

#Print evaluation metrics

print(f"Mean absolute error (MAE):{mae}")
print(f"Mean squared error (MSE):{mse}")
print(f"Root mean squared error (RMSE):{rmse}")
print(f"R² Score:{r2}")


# Initialize Decision Tree Regressor
dt_model=DecisionTreeRegressor(random_state=42)

#Train the model
dt_model.fit(X_train,y_train)

#Make Predictions
y_pred_dt=dt_model.predict(X_test)

#Evaluate Model
mae_dt=mean_absolute_error(y_test,y_pred_dt)
mse_dt=mean_squared_error(y_test,y_pred_dt)
rmse_dt=mean_squared_error(y_test,y_pred_dt) ** 0.5
r2_dt=r2_score(y_test,y_pred_dt)

#Print Results

print(f"Decision Tree - Mean Absolute Error:{mae_dt}")
print(f"Decision Tree - Mean Squared Error:{mse_dt}")
print(f"Decision Tree - Root Mean Squared Error:{rmse_dt}")
print(f"Decision Tree - R² Score:{r2_dt}")

#Train a Random Forest Model

#Initialize Random Forest regressor
rf_model=RandomForestRegressor(n_estimators=100,random_state=42)

#Train the model
rf_model.fit(X_train,y_train)

#Make predictions
y_pred_rf=rf_model.predict(X_test)

#Evaluate model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

#Print Results
print(f"\n Random Forest - Mean Absolute Erro:{mae_rf}")
print(f"Random Forest - Mean Squared Error:{mse_rf}")
print(f"Random Forest - Root Mean Squared Error:{rmse_rf}")
print(f"Random Forest - R² Score:{r2_rf}")

#Define parameter grid
param_grid_dt={
    "max_depth":[5,10,15,20],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,5]
}

#Initialize Decision Tree Regressor
dt=DecisionTreeRegressor()

#Perform Grid search
grid_dt=GridSearchCV(dt,param_grid_dt, cv=5, scoring="r2", n_jobs=-1)
grid_dt.fit(X_train,y_train)

#Best parameters & Score
print("Best parameters for decision tree:",grid_dt.best_params_)
print("Best R² Score:",grid_dt.best_score_)

#Train With best parameters
best_dt=grid_dt.best_estimator_

#Define parameter grid

param_grid_rf={
    "n_estimators":[50,100,200],
    "max_depth":[10,20,30],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,5]
}

#Initialize  Random forest
rf=RandomForestRegressor(random_state=42)

#Perform Randomized Search
random_search_rf=RandomizedSearchCV(rf, param_grid_rf, cv=5, scoring="r2", n_iter=10, n_jobs=-1,random_state=42)
random_search_rf.fit(X_train,y_train)

#Best parameter and score
print("Best parameters for Random Forest:",random_search_rf.best_params_)
print("Best  R² Score:",random_search_rf.best_score_)

#Train with best Parameters
best_rf=random_search_rf.best_estimator_

#Predict with best models
y_pred_best_dt=best_dt.predict(X_test)
y_pred_best_rf=best_rf.predict(X_test)

#Compute Metrices for Decision Tree

mae_dt=mean_absolute_error(y_test,y_pred_best_dt)
rmse_dt=mean_squared_error(y_test,y_pred_best_dt,squared=False)
r2_dt=r2_score(y_test,y_pred_best_dt)

#Compare Results
print("\nOptimize Decision Tree Performance:")
print(f"MAE:{mae_dt},RMSE:{rmse_dt},R² Score:{r2_dt}")

print("\nOptimize Random Forest Performance:")
print(f"MAE:{mae_rf},RMSE:{rmse_rf},R² Score:{r2_rf}")