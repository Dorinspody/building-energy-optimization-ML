import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score)



#LOAD DATASET
data = pd.read_excel("ENB2012_data.xlsx")
print("Dataset shape:", data.shape)
print("\nColumn names:")
print(data.columns)




# Building Parameters
X = data.iloc[:, 0:8]

# Last 2 columns:Y1 = Heating Load, Y2 = Cooling Load

y_heating = data.iloc[:, 8]
y_cooling = data.iloc[:, 9]

#SPLIT DATASET
X_train, X_test, y_heating_train, y_heating_test, \
y_cooling_train, y_cooling_test = train_test_split(X,y_heating,y_cooling,test_size=0.2,random_state=42)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



param_grid = {"n_estimators": [200,500,800],"max_depth": [5,10,15,None],"min_samples_split": [2,5,10],"min_samples_leaf": [1,2,4],"max_features": [0.5,1.0]}

#REATE RANDOM FOREST MODEL
base_model = RandomForestRegressor(random_state=42,n_jobs=-1)
print("\nSearching for the best model for HEATING LOAD...")
heating_grid_search = GridSearchCV(estimator=base_model,param_grid=param_grid,cv=5,scoring="neg_mean_squared_error",n_jobs=-1,verbose=1)
heating_grid_search.fit(X_train,y_heating_train)

# Best heating model
heating_model = heating_grid_search.best_estimator_
print("\nBest Heating Model Parameters:")
print(heating_grid_search.best_params_)


# GRID SEARCH FOR COOLING LOAD
print("\nSearching for the best model for COOLING LOAD...")
cooling_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42,n_jobs=-1),param_grid=param_grid,cv=5,scoring="neg_mean_squared_error",n_jobs=-1,verbose=1)

cooling_grid_search.fit(X_train,y_cooling_train)

# Best cooling model
cooling_model = cooling_grid_search.best_estimator_
import joblib
joblib.dump(heating_model,"heating_random_forest.pkl")
joblib.dump(cooling_model,"cooling_random_forest.pkl")
print("\nRandom Forest models saved successfully.")
print("\nBest Cooling Model Parameters:")
print(cooling_grid_search.best_params_)



#PREDICTIONS
heating_prediction = heating_model.predict(X_test)
cooling_prediction = cooling_model.predict(X_test)

#EVALUATION FUNCTION
def evaluate_model(y_true,y_prediction,model_name):
    mse = mean_squared_error(y_true, y_prediction)
    r2 = r2_score(y_true,y_prediction)
    rmse = np.sqrt(mean_squared_error(y_true,y_prediction))
    mae = mean_absolute_error(y_true,y_prediction)

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)
    print("R²:", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MSE:", mse)

#VALUATE BOTH MODELS


evaluate_model(y_heating_test,heating_prediction,"HEATING LOAD MODEL")
evaluate_model(y_cooling_test,cooling_prediction,"COOLING LOAD MODEL")


feature_importance = pd.DataFrame({"Feature": X.columns,"Heating Importance":heating_model.feature_importances_,"Cooling Importance":cooling_model.feature_importances_})
feature_importance = feature_importance.sort_values(by="Heating Importance",ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# TEST ONE BUILDING DESIGN

new_building = pd.DataFrame(
    [[
        0.80,      # Relative Compactness
        600,       # Surface Area
        300,       # Wall Area
        150,       # Roof Area
        7,         # Overall Height
        3,         # Orientation
        0.25,      # Glazing Area
        4          # Glazing Area Distribution
    ]],columns=X.columns)
predicted_heating = heating_model.predict(new_building)
predicted_cooling = cooling_model.predict(new_building)
print("\n" + "=" * 50)
print("NEW BUILDING PREDICTION")
print("=" * 50)
print("Predicted Heating Load:",predicted_heating[0])
print("Predicted Cooling Load:",predicted_cooling[0])


