// network.c
// Handle all the network
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/network.h"
#include "../include/neuron.h"
#include "../include/training.h"

XORNetwork* create_xor_network()
{

    XORNetwork* net = malloc(sizeof(XORNetwork));

    srand( time( NULL ) ); // Initalization for random

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            net->hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 0.5 - 0.25;
        }
        net->hidden_bias[i] = ((double)rand() / RAND_MAX) * 0.5 - 0.25;
        net->hidden_output[i] = 0.0;
    }
    
    for (int i = 0; i < 2; i++) {
        net->output_weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    net->output_bias = ((double)rand() / RAND_MAX) * 0.5 - 0.25;
    net->output = 0.0;

    return net;
}


void destroy_xor_network(XORNetwork* net)
{
    free(net);
}


double forward_pass(XORNetwork* net, double x1, double x2)
{
    double inputs[2] = {x1, x2};

    // Hidden layer
    double B1 = compute_neuron_output(inputs, net->hidden_weights[0],
                net->hidden_bias[0], 2);
    double B2 = compute_neuron_output(inputs, net->hidden_weights[1],
                net->hidden_bias[1], 2);

    double final_inputs[2] = {B1, B2};

    // Stockage
    net->hidden_output[0] = B1;
    net->hidden_output[1] = B2;

    double result = compute_neuron_output(final_inputs, net->output_weights,
            net->output_bias, 2);

    net->output = result;

    return result;
}


void print_network_state(XORNetwork* net)
{
    printf("=== XOR NETWORK STATE ===\n");
    
    // Entry layer
    printf("HIDDEN LAYER:\n");
    for (int i = 0; i < 2; i++) {
        printf("  Neuron H%d:\n", i+1);
        printf("    Weights: [%.3f, %.3f]\n", 
               net->hidden_weights[i][0], net->hidden_weights[i][1]);
        printf("    Bias: %.3f\n", net->hidden_bias[i]);
        printf("    Output: %.3f\n", net->hidden_output[i]);
    }
    
    // Output layer
    printf("OUTPUT LAYER:\n");
    printf("  Weights: [%.3f, %.3f]\n", 
           net->output_weights[0], net->output_weights[1]);
    printf("  Bias: %.3f\n", net->output_bias);
    printf("  Output: %.3f\n", net->output);
    
    // Stats
    printf("STATS:\n");
    printf("  Epochs: %d\n", net->training_epochs);
    printf("  Final cost: %.6f\n", net->final_cost);
    printf("========================\n");
}

void test_xor_network(XORNetwork* net)
{
    printf("XOR Network tests:\n");
    
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double expected[4] = {0, 1, 1, 0};
    
    for (int i = 0; i < 4; i++) {
        double result = forward_pass(net, inputs[i][0], inputs[i][1]);
        printf("  %.0f XOR %.0f = %.3f (expected: %.0f)\n", 
               inputs[i][0], inputs[i][1], result, expected[i]);
    }
    
    double cost = compute_network_cost(net);
    printf("  Average cost: %.6f\n", cost);
}
