# XOR Neural Network

Implementation of a neural network from scratch in C to solve the XOR problem. This project demonstrates the basics of multi-layer neural networks, backpropagation, and supervised learning.

## Architecture

- **Input Layer**: 2 nodes (A, B)
- **Hidden Layer**: 2 nodes with sigmoid activation
- **Output Layer**: 1 node with sigmoid activation
- **Learning Algorithm**: Backpropagation with gradient descent

## Project Structure

```
xor_neural_network/
├── src/
│   ├── main.c          # Main program and argument parsing
│   ├── neuron.c        # Basic neuron functions (sigmoid, compute_neuron_output)
│   ├── network.c       # Network structure and forward pass
│   └── training.c      # Training algorithm and backpropagation
├── include/
│   ├── neuron.h        # Neuron function declarations
│   ├── network.h       # Network structure definitions
│   └── training.h      # Training function declarations
├── Makefile
└── README.md
```

## Compilation

```bash
make
```

This creates the executable `xor_neural_network`.

To clean build files:
```bash
make clean
```

## Usage

### Basic usage:
```bash
./xor_neural_network
```

### With custom parameters:
```bash
./xor_neural_network [-e epochs] [-lr rate] [-v|-q] [-s filename]
```

**Options:**
- `-e epochs`: Number of training epochs (default: 10000)
- `-lr rate`: Learning rate (default: 0.5)
- `-v`: Verbose mode (show training progress)
- `-q`: Quiet mode (minimal output)
- `-s filename`: Save trained weights to file (default: xor_weights.dat)

### Examples:
```bash
# Train for 50,000 epochs with learning rate 0.1
./xor_neural_network -e 50000 -lr 0.1

# Silent training with custom save file
./xor_neural_network -q -s my_weights.dat

# Show help
./xor_neural_network -h
```

## Expected Output

**Before training:**
```
XOR Network tests:
  0 XOR 0 = 0.593 (expected: 0)
  0 XOR 1 = 0.245 (expected: 1)
  1 XOR 0 = 0.738 (expected: 1)
  1 XOR 1 = 0.421 (expected: 0)
```

**After successful training:**
```
XOR Network tests:
  0 XOR 0 = 0.005 (expected: 0)
  0 XOR 1 = 0.996 (expected: 1)
  1 XOR 0 = 0.995 (expected: 1)
  1 XOR 1 = 0.004 (expected: 0)
```

## Implementation Details

### Network Structure
```c
typedef struct {
    double hidden_weights[2][2];  // Input to hidden weights
    double hidden_bias[2];        // Hidden layer biases
    double hidden_output[2];      // Hidden layer outputs
    
    double output_weights[2];     // Hidden to output weights
    double output_bias;           // Output bias
    double output;                // Final output
} XORNetwork;
```

### Key Functions
- `sigmoid(x)`: Sigmoid activation function
- `create_xor_network()`: Initialize network with random weights
- `forward_pass()`: Calculate network output for given inputs
- `train_single_example()`: Update weights using backpropagation
- `compute_network_cost()`: Calculate total error across all XOR cases

### Training Algorithm
1. **Forward Pass**: Calculate network output
2. **Error Calculation**: Compare with expected output
3. **Backpropagation**: 
   - Calculate output layer error
   - Propagate error back to hidden layer
   - Update all weights proportionally to their contribution to the error

## Performance Notes

- **Success Rate**: ~90% of runs converge to correct solution
- **Convergence**: Usually within 10,000-50,000 epochs
- **Learning Rate**: 0.1-0.5 works well; higher values may cause instability
- **Random Initialization**: Different runs may have different convergence rates

If a run fails to converge (outputs stuck around 0.5), simply run again with different random initialization.

## Mathematical Foundation

The XOR problem is **non-linearly separable**, meaning it cannot be solved with a single-layer perceptron. This network uses:

- **Hidden Layer**: Each neuron learns to separate the input space with different lines
- **Output Layer**: Combines hidden layer outputs to create the correct XOR boundary
- **Backpropagation**: Adjusts weights by propagating errors backward through the network

## Learning Objectives

This implementation demonstrates:
- Multi-layer neural network architecture
- Backpropagation algorithm
- Gradient descent optimization
- The importance of hidden layers for non-linear problems
- Weight initialization and convergence issues

## Next Steps

This basic implementation can be extended to:
- Image recognition (digit classification, letter recognition)
- More complex logical operations
- Different activation functions (ReLU, tanh)
- Advanced optimizers (momentum, Adam)

## Dependencies

- GCC compiler
- Math library (`-lm` flag)
- Standard C libraries

No external dependencies required.
