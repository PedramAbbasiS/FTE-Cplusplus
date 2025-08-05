# FTE-Cplusplus
Function derivative calculator

# How to run 
1. Please open a terminal like git bash and run the following address
    git clone https://github.com/PedramAbbasiS/FTE-Cplusplus.git

2. Open visual studio (or the IDE of your choice) and click on "open a local folder" and open the folder that you just cloned

3. Switch to FTE-Cplusplus branch

4. Click on derivative calculator folder in the solution explorer and click on calculator.cpp

5. Go to build, build the current document on the top down menu on the top and click on it

6. You can run or debug the code, and modify the value of x0, the point at which the derivative of the function is calculated at the begining of the main()

Please note this project requires a C++14- compatible compiler
The description about the code will be found in the rest of this document as well as in the code comment sections in the code.

# Code structure: 
Forward and Backward Automatic Differentiation

This project implements a calculator for evaluating mathematical functions and their derivatives using both forward and backward (reverse-mode) automatic differentiation 
in C++14. 

# Features

- Forward Mode:
  Uses a polymorphic design ("function" class and its derived classes) to represent and evaluate functions and their derivatives.
- The forward mode uses only one forward pass in which it caclculates values and derivatives simultaneously at function levels.  
  
- Backward Mode: 
  Uses a computational graph ("node" and its derived classes) to propagate gradients for reverse-mode differentiation.
- backward mode uses a two-pass approach, first to evaluate the function and then to compute the gradients which makes it for this case less efficient than forward mode.
- I put this here only for demonstration purposes but this method could be usefull when the function is very complex and has many variables.

- function definiton features
  - Addition, subtraction, multiplication, division
  - Power and logarithm functions
  - Constant and input nodes

# Error Handling
- The program checks for invalid input (such as log of non-positive values, division by zero) and exits with an error message.
