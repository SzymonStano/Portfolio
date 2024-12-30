# Neural Network Solution for an Ordinary Differential Equation

This project demonstrates the use of neural networks to approximate the solution to a first-order ordinary differential equation (ODE):

$$
\begin{cases}
y' + y = 0, \\
y(0) = 1.
\end{cases}
$$

The exact analytical solution of this ODE is \(y = e^{-t}\).

## Methodology
*This methodology can be applied to any ODE in residual form.*


The neural network approach involves the following steps:

1. **Input and Output Design**:  
   The neural network takes a time interval as input and is trained to output a function that approximates the solution to the ODE over that interval.

2. **Reformulating the ODE**:  
   The given ODE is expressed in its residual form:  
   \[
   \text{Residual}(t, y) = y'(t) + y(t).
   \]
   A valid solution must satisfy \(\text{Residual}(t, y) = 0\) and initial condition \(y(0) = 1\).

3. **Automatic Differentiation**:  
   Derivatives required for the residual calculation are computed using automatic differentiation, a powerful feature of modern deep learning frameworks.

4. **Constructing the Loss Function**:  
   A custom loss function is built based on the residual of the ODE and its initial condition. During training, the neural network minimizes the squared residual over a set of training points, thus satisfying the given differential equation:  
   \[
   \text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (\text{Residual}(t_i, y_i))^2 + (y(0)-1)^2.
   \]

5. **Training the Neural Network**:  
   The model is trained using different activation functions (`tanh` and `ReLU`) to compare their performance in approximating the solution.

6. **Evaluating the Model**:  
   The performance of the models is evaluated based on:  
   - Mean Squared Error (MSE) compared to the analytical solution.
   - Absolute and relative tolerances between the predicted and analytical solutions.



## Contents

- `Solution.ipynb`: The main Jupyter Notebook containing the implementation of the methodology, hyperparameter tuning, and analysis.
- `Imports/`: A folder with supporting modules:
  - `HP_Tuning.py`: Implements hyperparameter tuning strategies.
  - `Models.py`: Contains classes and functions for building and training neural network models.
  - `Loss_functions.py`: Defines the residual-based loss function and utilities for derivative computations using automatic differentiation.
  - `Solution_precision.py`: Tools for calculating absolute tolerance.
- `README.md`: Project documentation (this file).

## Key Findings

- The model with `tanh` activation function required fewer resources (12 layers, 61 neurons per layer) and shorter training time (~13.57s) compared to the model with `ReLU` (14 layers, 207 neurons per layer, ~46.79s).
- The `tanh` model demonstrated better performance with lower MSE and absolute tolerance, making it more effective for solving ODEs.
- Automatic differentiation allowed precise computation of derivatives, contributing to the accuracy of the residual evaluation and overall model performance, but might not be suitable for higher-order ODEs.

