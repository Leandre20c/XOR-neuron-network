#ifndef NETWORK_H
#define NETWORK_H

// ===============================
// STRUCTURE DU RÉSEAU XOR
// ===============================

typedef struct {
    // Couche cachée (2 neurones)
    double hidden_weights[2][2];  // [neurone][entrée]
    double hidden_bias[2];        // [neurone]
    double hidden_output[2];      // Activations courantes
    
    // Couche sortie (1 neurone)
    double output_weights[2];     // [entrée_cachée]
    double output_bias;
    double output;                // Activation courante
    
    // Statistiques d'entraînement
    int training_epochs;
    double final_cost;
} XORNetwork;

/**
 * Crée et initialise un nouveau réseau XOR
 * Les poids sont initialisés aléatoirement
 */
XORNetwork* create_xor_network(void);

/**
 * Libère la mémoire du réseau
 */
void destroy_xor_network(XORNetwork* net);

/**
 * Propagation avant: calcule la sortie du réseau
 * net: pointeur vers le réseau
 * x1, x2: entrées (0 ou 1 pour XOR)
 * retourne: sortie du réseau [0,1]
 */
double forward_pass(XORNetwork* net, double x1, double x2);

/**
 * Affiche l'état complet du réseau (pour debugging)
 */
void print_network_state(XORNetwork* net);

/**
 * Teste le réseau sur les 4 cas XOR
 */
void test_xor_network(XORNetwork* net);

#endif
