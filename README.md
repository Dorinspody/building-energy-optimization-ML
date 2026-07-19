Building Energy Optimization with Machine Learning

This project develops a machine-learning-based framework for optimizing building energy performance using the UCI Energy Efficiency dataset.

Random Forest Modeling

The dataset contains eight building design parameters (X1–X8) and two target variables: Heating Load (Y1) and Cooling Load (Y2).

Two separate Random Forest Regression models are trained:

* One model predicts Heating Load.
* One model predicts Cooling Load.

GridSearchCV is used to search for suitable Random Forest hyperparameters. The trained models are then saved and used as surrogate models during the optimization stage.

Model Performance

Heating Load Model

* R²: 0.9977
* RMSE: 0.4869
* MAE: 0.3509
* MSE: 0.2371

Cooling Load Model

* R²: 0.9689
* RMSE: 1.6978
* MAE: 1.0628
* MSE: 2.8824

The results show that the Random Forest models can accurately approximate the relationship between building design parameters and energy loads.

Single-Objective Optimization: Differential Evolution

After training, the Random Forest models are used as surrogate models. Differential Evolution searches through possible building designs and uses the trained models to predict their energy performance.

The single objective is defined as:


Total Load = Heating Load + Cooling Load


The optimization process searches for the building design that minimizes this total predicted energy load.

The optimized design produced:

* Heating Load: 6.03496
* Cooling Load: 11.05088
* Total Load: 17.08584

Differential Evolution is used because the problem is nonlinear and the Random Forest model does not provide a differentiable mathematical function. Therefore, a gradient-free evolutionary optimization method is suitable.

Multi-Objective Optimization: NSGA-II

A second optimization approach is implemented using NSGA-II.

Unlike Differential Evolution, NSGA-II does not combine the two energy loads into a single objective. Instead, it optimizes them independently:


Minimize Heating Load
Minimize Cooling Load

This produces a set of Pareto-optimal building designs. Each design represents a different trade-off between heating and cooling energy requirements.

For example, one design may achieve a very low heating load but a slightly higher cooling load, while another may reduce cooling load more effectively.

The resulting Pareto front allows engineers to select a building design according to their specific energy priorities.

Why Use Both Optimization Methods?

Differential Evolution and NSGA-II solve different versions of the optimization problem.

Differential Evolution answers:


"Which single building design minimizes the combined total energy load?"


NSGA-II answers:

"Which building designs provide the best trade-offs between heating and cooling loads?"

Using both methods provides a more complete analysis.

The single-objective optimization provides one clear design with the lowest combined predicted load, while the multi-objective optimization provides a range of efficient alternatives.

In this project, both optimization approaches identified the same key optimum:

* Heating Load: approximately 6.03
* Cooling Load: approximately 11.05

This agreement increases confidence that the solution is not simply an artifact of one optimization algorithm.

However, the two methods provide different types of information. Differential Evolution gives a single best solution based on the combined objective, while NSGA-II provides the complete trade-off between the two competing objectives.


This project demonstrates the use of machine learning surrogate models and evolutionary optimization for energy-efficient building design.









<img width="852" height="607" alt="Minimized heating load and cooling " src="https://github.com/user-attachments/assets/90d7a66f-6520-404a-91d1-bdbf428a2ee9" />
