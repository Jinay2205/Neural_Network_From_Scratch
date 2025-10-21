// Neural Network Class declarations.

#pragma once

#include <vector>
using namespace std;

// declaring the Neural Network class
class NeuralNetwork
{
public:
    // Constructor to initialize various variables.
    NeuralNetwork(int input_size, int hidden_size, int output_size);

    // Function to update the weights and bias of each layer using RMSProp optimization.
    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, double learining_rate, int epochs);

    // Function to predict the output by forward feeding the testing inputs.
    vector<double> feedForward(const vector<double> &input);

private:
    // Declaring layer sizes.
    int input_size;
    int hidden_size;
    int output_size;

    // Declaring layer weights and bias.
    vector<vector<double>> weights_input_hidden;
    vector<vector<double>> weights_hidden_output;
    vector<double> bias_hidden;
    vector<double> bias_output;

    // Declaring the RMSProp cache variables to store the moving average of squared gradients.
    vector<vector<double>> rmsprop_cache_input_hidden;
    vector<vector<double>> rmsprop_cache_hidden_output;
    vector<double> rmsprop_cache_bias_hidden;
    vector<double> rmsprop_cache_bias_output;
};