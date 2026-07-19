Building Energy Optimization

This project uses machine learning and evolutionary optimization to improve building energy efficiency.

The UCI Energy Efficiency dataset is used to analyze building design parameters.

Eight building parameters are used as input features, including compactness, surface area, wall area, roof area, height, orientation, and glazing characteristics.

Two Random Forest regression models are trained to predict heating and cooling loads.

GridSearchCV is used to find suitable Random Forest hyperparameters.

The trained models are evaluated using MSE, RMSE, MAE, and R² metrics.

The Random Forest models are then saved and used as surrogate models for optimization.

Differential Evolution searches for building parameters that minimize the total predicted energy load.
the parameters constraint:

The objective function is defined as the sum of the predicted heating and cooling loads.

A multi-objective optimization approach using NSGA-II is also implemented.

NSGA-II optimizes heating and cooling loads separately and generates a Pareto front.

The Pareto front represents different trade-offs between heating and cooling energy requirements.

The optimized building parameters can be compared with the original dataset designs.

This workflow demonstrates how machine learning can support energy-efficient building design.

The project combines predictive modeling, surrogate modeling, and evolutionary optimization.
