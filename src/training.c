#include "../include/training.h"
#include "../include/neuron.h"
#include "../include/network.h"
#include <stdio.h>


void train_single_example(XORNetwork* net, XORExample example, double
        learning_rate)
{
    double A = example.input[0];
    double B = example.input[1];

    // Forward pass
    double res = forward_pass(net, A, B);

    // Calcul of exit error
    double error = example.expected - res;

    // == Back propagation ==

    // Exit layer delta
    double output_error = error * sigmoid_derivative(net->output);

    // Save old weigths
    double old_output_weights[2];
    old_output_weights[0] = net->output_weights[0];
    old_output_weights[1] = net->output_weights[1];

    // Ajust output weigts
    net->output_weights[0] += learning_rate * output_error * net->hidden_output[0];
    net->output_weights[1] += learning_rate * output_error * net->hidden_output[1];
    net->output_bias += learning_rate * output_error * 1.0;

    
    // == Ajust Hidden Layer

    double hidden_error_1 = output_error * old_output_weights[0] *
        sigmoid_derivative(net->hidden_output[0]);

    double hidden_error_2 = output_error * old_output_weights[1] *
        sigmoid_derivative(net->hidden_output[1]); 

    // neuron 1
    net->hidden_weights[0][0] += learning_rate * hidden_error_1 * A;  // w to A
    net->hidden_weights[0][1] += learning_rate * hidden_error_1 * B;  // w to B
    net->hidden_bias[0] += learning_rate * hidden_error_1;

    // neuron 2
    net->hidden_weights[1][0] += learning_rate * hidden_error_2 * A;  // w to A
    net->hidden_weights[1][1] += learning_rate * hidden_error_2 * B;  // w to B
    net->hidden_bias[1] += learning_rate * hidden_error_2;
}


void train_xor_network(XORNetwork* net, int epochs, double learning_rate,
        int verbose)
{
    double input_map[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double output_map[4] = {0, 1, 1, 0};

    XORExample example;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < 4; i++)
        {
            example.input[0] = input_map[i][0];
            example.input[1] = input_map[i][1];

            example.expected = output_map[i]; 

            train_single_example(net, example, learning_rate);
        }

        if (verbose && epoch % 1000 == 0) {
            double cost = compute_network_cost(net);
            printf("Epoch %d, Cost: %.6f\n", epoch, cost);
        }
    }

     
}


double compute_network_cost(XORNetwork* net)
{
    double total_cost = 0.0;
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double expected[4] = {0, 1, 1, 0};

    // Test all cases
    for (int i = 0; i < 4; i++) {
        double prediction = forward_pass(net, inputs[i][0], inputs[i][1]);
        double error = expected[i] - prediction;
        total_cost += error * error;  
    }

    // Overall
    return total_cost / 4.0;
    
}


void save_network(XORNetwork* net, const char* filename)
{
    FILE* file = fopen(filename, "wb");
    if (file == NULL) return;
    
    fwrite(net, sizeof(XORNetwork), 1, file);
    fclose(file);
}

int load_network(XORNetwork* net, const char* filename)
{
    FILE* file = fopen(filename, "rb");
    if (file == NULL) return 0;
    
    fread(net, sizeof(XORNetwork), 1, file);
    fclose(file);
    return 1;
}
