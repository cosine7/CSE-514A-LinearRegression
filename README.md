# CSE-514A-LinearRegression

## Required Package
- pandas
- numpy
- math
- matplotlib
- xlrd

Please make sure above packages are installed. Also, you should configure your own venv (Python Virtual Environment)

## How to run the program?

After all packages are installed and the environment is configured, just run the main.py file

In main.py, data is divided by four type:

- training set
- testing set
- normalized training set
- normalized testing set

The program would run all above four datasets for

- univariate model
- multivariate model
- quadratic model

Corresponding variance explained and coefficients would be printed on console during runtime. Univariate plots would be stored in the results file as well.

## Extra Stuff

The normalization method used is mean normalization. The normalization function is under algorithm/utility.py, and the quadratic model and all other models are under algorithm folder