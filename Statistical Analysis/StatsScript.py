import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Author: Zachary Kirkeby

file_path = "<File_Path>"  # Replace with your local file path for non zach users
data = pd.read_csv(file_path, header=None)  # header=none ensures line 1 isn't skipped
data.columns = ['x', 'y', 'z']  # rep linear acceleration

x_values = data['x']
y_values = data['y']
z_values = data['z']


# use ~10 for smoothing
# Data Plotting
def plot_data(x_values, y_values, z_values):
    fig, ax = plt.subplots()  # Create one figure and one set of subplots
    time = np.linspace(0, 322, 322)
    ax.plot(time, x_values, label='x vals')
    ax.plot(time, y_values, label='quadratic')
    ax.plot(time, z_values, label='cubic')
    ax.set_xlabel('Time (units tbd)')
    ax.set_ylabel('y label name')
    ax.set_title(" Vars X v Y v Z in respect to time")
    ax.legend()
    plt.show()  # Display the plot using plt.show() at the end
    plt.savefig('DataPlot.png')
    # saves the file
    # leave commented out while testing


def get_covariance(x_values, y_values, z_values):
    data_matrix = np.vstack((x_values, y_values, z_values))
    covariance_matrix = np.cov(data_matrix)
    print("Covariance Matrix:")
    print(covariance_matrix)



def get_statistics(x_values, y_values, z_values):
    # high level stats
    print("Mean of datasets")
    mean_x = x_values.mean()
    mean_y = y_values.mean()
    mean_z = z_values.mean()
    print("x: " + str(mean_x))
    print("y: " + str(mean_y))
    print("z: " + str(mean_z))

    print("Standard Deviations of datasets")
    std_x = x_values.std()
    std_y = y_values.std()
    std_z = z_values.std()
    print("x: " + str(std_x))
    print("y: " + str(std_y))
    print("z: " + str(std_z))

    print("Kurtosis of datasets")
    kurt_x = x_values.kurtosis()
    kurt_y = y_values.kurtosis()
    kurt_z = z_values.kurtosis()
    print("x: " + str(kurt_x))
    print("y: " + str(kurt_y))
    print("z: " + str(kurt_z))

    quantiles_x = np.percentile(x_values, [25, 50, 75])
    quantiles_y = np.percentile(y_values, [25, 50, 75])
    quantiles_z = np.percentile(z_values, [25, 50, 75])

    print("\nQuantiles:")
    print(quantiles_x)
    print(quantiles_y)
    print(quantiles_z)


def get_moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def graph_moving_value(x_sma, y_sma, z_sma):
    fig, ax = plt.subplots()  # Create one figure and one set of subplots
    time = np.linspace(0, 293, 293)
    ax.plot(time, x_sma, label='x vals')
    ax.plot(time, y_sma, label='y vals')
    ax.plot(time, z_sma, label='z vals')
    ax.set_xlabel('Time (units tbd)')
    ax.set_ylabel('y label name')
    ax.set_title(" Vars X v Y v Z in respect to time")
    ax.legend()
    plt.show()  # Display the plot using plt.show() at the end
    plt.savefig('MovingAveragePlot')
    # saves the file
    # leave commented out while testing


def get_linear_regression(x_value, z_value):
    # make dots smaller
    mean_x = x_value.mean()
    mean_z = z_value.mean()
    numerator = np.sum((x_value - mean_x) * (z_value - mean_z))
    denominator = np.sum((x_value - mean_x) ** 2)
    m = numerator / denominator
    c = mean_z - m * mean_x
    z_pred = m * x_values + c
    plt.scatter(x_value, z_value, color='red', label='Data Points')
    plt.plot(x_value, z_pred, color='blue', linewidth=3, label='Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.show()
    plt.savefig('LinearRegressionPlot')


def get_polynomial_regression(x_value, z_value, degree):
    coefficients = np.polyfit(x_value, z_value, degree)
    poly_func = np.poly1d(coefficients)
    # curve smoothing with a range
    x_range = np.linspace(min(x_value), max(x_value), 100)
    z_pred = poly_func(x_range)
    # Plot the results
    plt.scatter(x_value, z_value, color='red', label='Data Points')
    plt.plot(x_range, z_pred, color='blue', linewidth=3, label=f'Polynomial Regression (Degree {degree})')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.show()
    plt.savefig('PolynomialRegressionPlot')

def get_Least_Squares():
    model = ols('x~y', data=data).fit()
    print(model.summary())
    model = ols('y~z', data=data).fit()
    print(model.summary())
    model = ols('x~z', data=data).fit()
    print(model.summary())
    fig = plt.figure(figsize=(20, 10))

def make_regression_plot(var):
    fig = sm.graphics.plot_regress_exog(model, var, fig=fig)
    fig.show()

#******************************************* Run Code *******************************************
plot_data(x_values,y_values,z_values)
get_covariance(x_values, y_values, z_values)
get_statistics(x_values, y_values, z_values)
moving_x = get_moving_average(x_values,5)
moving_y = get_moving_average(y_values,5)
moving_z = get_moving_average(z_values,5)
# stick to 10-20 tops
#scipy

graph_moving_value(moving_x, moving_y, moving_z)
get_linear_regression(x_values,z_values)
get_polynomial_regression(x_values, z_values, 2)
get_polynomial_regression(x_values, z_values, 3)
get_polynomial_regression(x_values, z_values, 4)

get_linear_regression(moving_x,moving_z)
get_polynomial_regression(moving_x, moving_z, 2)
get_polynomial_regression(moving_x, moving_z, 3)
get_polynomial_regression(moving_x, moving_z, 4)

get_least_squared()

make_regression_plot(x_values)
make_regression_plot(y_values)
make_regression_plot(z_values)
