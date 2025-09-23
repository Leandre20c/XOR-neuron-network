#include "../include/neuron.h"
#include "../include/network.h"
#include "../include/training.h"
#include <stdlib.h>
#include <stdio.h>




int main()
{
    printf("\nCreating Neuron Network\n\n");
    XORNetwork* xor_network = create_xor_network();

    printf("\nBefore training\n");
    test_xor_network(xor_network);

    printf("\nTraining network...\n");
    train_xor_network(xor_network, 10000, 0.5, 1);

    printf("\nTest after training\n");
    test_xor_network(xor_network);

    destroy_xor_network(xor_network);

    printf("\nEnd\n");
    return 0;
}

