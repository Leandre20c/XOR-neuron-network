#ifndef NEURON_H
#define NEURON_H

// ===============================
// FONCTIONS MATHÉMATIQUES DE BASE
// ===============================

/**
 * Fonction sigmoid: transforme n'importe quel nombre en [0,1]
 * Utilisée comme fonction d'activation des neurones
 */
double sigmoid(double x);

/**
 * Dérivée de sigmoid (nécessaire pour l'apprentissage)
 */
double sigmoid_derivative(double x);

/**
 * Calcule la sortie d'un neurone simple
 * inputs: tableau des entrées
 * weights: tableau des poids
 * bias: biais du neurone
 * n: nombre d'entrées
 */
double compute_neuron_output(double* inputs, double* weights,
                             double bias, int n);

/**
 * Affiche les détails du calcul d'un neurone (pour debugging)
 */
void debug_neuron(double* inputs, double* weights, double bias, int n);

#endif
