import joblib
import numpy as np

from scipy.optimize import differential_evolution

#multi-objective optimization using NSGA-II

# LOAD TRAINED RANDOM FOREST MODELS

heating_model = joblib.load("heating_random_forest.pkl")

cooling_model = joblib.load("cooling_random_forest.pkl")


# OBJECTIVE FUNCTION
def objective_function(parameters):

    parameters = np.array(parameters,dtype=float)
    parameters[5] = round(parameters[5])# X6: Orientation
    parameters[7] = round(parameters[7])# X8: Glazing Distribution
    heating_load = heating_model.predict([parameters])[0] # Heating prediction
    cooling_load = cooling_model.predict([parameters])[0]  # Cooling prediction

    # Objective function
    total_load = (heating_load +cooling_load)
    return total_load

#constraint
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

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


#load trained random forest models
heating_model = joblib.load("heating_random_forest.pkl")
cooling_model = joblib.load("cooling_random_forest.pkl")



feature_names = ["X1","X2","X3","X4","X5","X6","X7","X8"]


#DEFINE MULTI-OBJECTIVE PROBLEM
class BuildingOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=8,n_obj=2,n_ieq_constr=0,xl=np.array([0.62,514.5,245.0,110.25,3.5,2,0.0,0]),
            xu=np.array([0.98,808.5,416.5,220.5,7.0,5,0.4,5]))

    def _evaluate(self,x,out,*args,**kwargs):
        x = np.array(x,dtype=float)
        x[5] = round(x[5])
        x[7] = round(x[7])
        input_data = pd.DataFrame([x],columns=feature_names)
        heating_load = (heating_model.predict(input_data)[0])
        cooling_load = (cooling_model.predict(input_data)[0])
        out["F"] = [heating_load,cooling_load]


algorithm = NSGA2(pop_size=100)

#RUN OPTIMIZATION
problem = BuildingOptimizationProblem()
termination = get_termination("n_gen",200)

result = minimize(problem,algorithm,termination,seed=42,verbose=True)
optimal_parameters = result.X
objective_values = result.F


#PRINT PARETO SOLUTIONS
print("\n")
print("=" * 70)
print("PARETO-OPTIMAL BUILDING DESIGNS")
print("=" * 70)

for i in range(len(optimal_parameters)):
    parameters = (optimal_parameters[i]).copy()

    parameters[5] = round(parameters[5])
    parameters[7] = round(parameters[7])
    print("\n")
    print(f"Design {i + 1}")
    print("-" * 40)
    print("X1 - Relative Compactness:",parameters[0])
    print("X2 - Surface Area:",parameters[1])
    print("X3 - Wall Area:",parameters[2])
    print("X4 - Roof Area:",parameters[3])
    print("X5 - Overall Height:",parameters[4])
    print("X6 - Orientation:",parameters[5])
    print("X7 - Glazing Area:",parameters[6])
    print("X8 - Glazing Distribution:",parameters[7])
    print("Heating Load:",objective_values[i][0])
    print("Cooling Load:",objective_values[i][1])


# PLOT PARETO FRONT

plt.figure(figsize=(7, 5))

plt.scatter(objective_values[:, 0],objective_values[:, 1])
plt.xlabel("Heating Load")
plt.ylabel("Cooling Load")
plt.title("Pareto Front")
plt.grid(True)
plt.show()