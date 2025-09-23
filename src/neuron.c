#include "../include/neuron.h"
#include <math.h>


double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}


double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}


double compute_neuron_output(double* inputs, double* weights,
                             double bias, int n)
{
    double s = bias;

    for (int i = 0; i < n; i++)
    {
        s += inputs[i]*weights[i];
    }

    double output = sigmoid(s);

    return output;
}

