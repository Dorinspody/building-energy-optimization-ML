import joblib
import numpy as np
from scipy.optimize import differential_evolution



# LOAD TRAINED RANDOM FOREST MODELS
heating_model = joblib.load("heating_random_forest.pkl")
cooling_model = joblib.load("cooling_random_forest.pkl")

# OBJECTIVE FUNCTION
def objective_function(parameters):
    parameters = np.array(parameters,dtype=float)
    # X6: Orientation must be integer
    parameters[5] = round(parameters[5])
    # X8: Glazing Distribution must be integer
    parameters[7] = round(parameters[7])
    # Heating prediction
    heating_load = heating_model.predict([parameters])[0]
    # Cooling prediction
    cooling_load = cooling_model.predict([parameters])[0]
    # Objective function
    total_load = (heating_load +cooling_load)
    return total_load
bounds = [
    (0.62, 0.98),       # X1
    (514.5, 808.5),     # X2
    (245.0, 416.5),     # X3
    (110.25, 220.5),    # X4
    (3.5, 7.0),         # X5
    (2, 5),             # X6
    (0.0, 0.4),         # X7
    (0, 5)              # X8
]

result = differential_evolution(objective_function,bounds=bounds,strategy="best1bin",maxiter=1000,popsize=15,tol=1e-7,seed=42,polish=True,disp=True)
optimal_parameters = result.x
optimal_parameters[5] = round(optimal_parameters[5])
optimal_parameters[7] = round(optimal_parameters[7])
optimal_heating = heating_model.predict([optimal_parameters])[0]
optimal_cooling = cooling_model.predict([optimal_parameters])[0]
optimal_total = (optimal_heating +optimal_cooling)
print("\nOptimal Building Parameters:")
print(optimal_parameters)

print("\nPredicted Heating Load:")
print(optimal_heating)

print("\nPredicted Cooling Load:")
print(optimal_cooling)

print("\nPredicted Total Load:")
print(optimal_total)