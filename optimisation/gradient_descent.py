import sys

#define the objective function here
def objective_function(x):
    return x**2

def approximate_derivative(x):
    delta=0.001
    current_value=objective_function(x)
    next_value=objective_function(x+delta)
    derivative_value=(next_value-current_value)/delta
    return derivative_value

def gradient_descent(learning_rate):
    current_point=1.0
    value_delta=1
    while(value_delta>1e-6):
        historical_value=objective_function(current_point)
        current_point=current_point-learning_rate*approximate_derivative(current_point)
        current_value=objective_function(current_point)
        value_delta=abs(current_value-historical_value)
        print('moving to point '+str(current_point)+str( ' with value ')+str(current_value))
    return current_value

if __name__=='__main__':
    print('minimum found:' +str(gradient_descent(float(sys.argv[1]))))

