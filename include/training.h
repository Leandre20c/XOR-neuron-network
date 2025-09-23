#ifndef TRAINING_H
#define TRAINING_H

#include "network.h"

// ===============================
// DONNÉES ET APPRENTISSAGE
// ===============================

typedef struct {
    double input[2];    // x1, x2
    double expected;    // sortie attendue
} XORExample;

/**
 * Données d'entraînement XOR (les 4 cas possibles)
 */
extern XORExample XOR_DATA[4];

/**
 * Entraîne le réseau sur un seul exemple
 * net: réseau à entraîner
 * example: exemple d'entraînement
 * learning_rate: vitesse d'apprentissage (ex: 0.5)
 */
void train_single_example(XORNetwork* net, XORExample example, double learning_rate);

/**
 * Entraîne le réseau sur tous les exemples XOR
 * net: réseau à entraîner
 * epochs: nombre d'itérations d'entraînement
 * learning_rate: vitesse d'apprentissage
 * verbose: afficher les détails (1) ou non (0)
 */
void train_xor_network(XORNetwork* net, int epochs, double learning_rate, int verbose);

/**
 * Calcule le coût (erreur) du réseau sur tous les exemples
 */
double compute_network_cost(XORNetwork* net);

/**
 * Sauvegarde les poids du réseau dans un fichier
 */
void save_network(XORNetwork* net, const char* filename);

/**
 * Charge les poids depuis un fichier
 */
int load_network(XORNetwork* net, const char* filename);

#endif
