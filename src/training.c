#include "../include/training.h"
#include "../include/neuron.h"
#include "../include/network.h"
#include <stdio.h>


void train_single_example(XORNetwork* net, XORExample example, double
        learning_rate)
{
    double A = example.input[0];
    double B = example.input[1];
    double res = forward_pass(net, A, B);

    double error = example.expected - res;

    // Ajust output weigts
    net->output_weights[0] += learning_rate * error * net->hidden_output[0];
    net->output_weights[1] += learning_rate * error * net->hidden_output[1];
    net->output_bias += learning_rate * error * 1.0;

    // Ajust hidden layer
    double hidden_error_1 = error * net->output_weights[0];
    double hidden_error_2 = error * net->output_weights[1];

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
            printf("Époque %d, Coût: %.6f\n", epoch, cost);
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
    if (file == NULL) return 0;  // Échec
    
    fread(net, sizeof(XORNetwork), 1, file);
    fclose(file);
    return 1;  // Succès
}
