// Implementation of the Neural Network functions.

// Importing the header files containing declarations.
#include "NeuralNet.h"
#include "Utils.h"

// Importing libraries.
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

// Constructor for the NeuralNetwork class for initializing variables.
NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size)
{
    // Initialize weights and bias with zeros.
    weights_input_hidden.resize(input_size, vector<double>(hidden_size, 0.0));
    weights_hidden_output.resize(hidden_size, vector<double>(output_size, 0.0));
    bias_hidden.resize(hidden_size, 0.0);
    bias_output.resize(output_size, 0.0);

    // Initialize weights using Xavier initialization.
    double xavier_scale_input_hidden = sqrt(2.0 / (input_size + hidden_size));
    double xavier_scale_hidden_output = sqrt(2.0 / (hidden_size + output_size));

    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < hidden_size; j++)
        {
            // Weights of the input layer.
            weights_input_hidden[i][j] = (static_cast<double>(rand()) / RAND_MAX) * 2.0 * xavier_scale_input_hidden - xavier_scale_input_hidden;
        }
    }
    for (int i = 0; i < hidden_size; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            // Weights of the output layer.
            weights_hidden_output[i][j] = (static_cast<double>(rand()) / RAND_MAX) * 2.0 * xavier_scale_hidden_output - xavier_scale_hidden_output;
        }
    }

    // Initialize random bias.
    for (int i = 0; i < hidden_size; i++)
    {
        // Hidden layer bias.
        bias_hidden[i] = (rand() % 2000 - 1000) / 1000.0;
    }
    for (int i = 0; i < output_size; i++)
    {
        // Output layer bias.
        bias_output[i] = (rand() % 2000 - 1000) / 1000.0;
    }

    // Initialize RMSProp cache variables to zeros.
    // RMSProp cache variables are used to store the moving averages of the squared gradients (change in loss).
    rmsprop_cache_input_hidden.resize(input_size, vector<double>(hidden_size, 0.0));
    rmsprop_cache_hidden_output.resize(hidden_size, vector<double>(output_size, 0.0));
    rmsprop_cache_bias_hidden.resize(hidden_size, 0.0);
    rmsprop_cache_bias_output.resize(output_size, 0.0);
}

// Training function that updates the weights and bias of the layers based on input.
void NeuralNetwork::train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, double learning_rate, int epochs)
{
    // Loop through the number of epochs
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        // initialize the weight gradient (change in loss).
        double gradient_weight = 0.0;

        // Using size_t as it is an unsigned integer meant to represent size or indices.
        // Looping through the input size.
        for (size_t instance = 0; instance < inputs.size(); instance++)
        {
            // Forward pass:

            // assign the input and target vectors to the current rows.
            vector<double> input = inputs[instance];
            vector<double> target = targets[instance];

            // assign hidden layer and output layer outputs to zeros.
            vector<double> hidden(hidden_size, 0.0);
            vector<double> output(output_size, 0.0);

            // y_i = sum(w_ij*x_ij) + b_i;

            // For hidden layer.
            for (int i = 0; i < hidden_size; i++)
            {
                // Summation operation.
                for (int j = 0; j < input_size; j++)
                {
                    hidden[i] += input[j] * weights_input_hidden[j][i];
                }
                hidden[i] += bias_hidden[i];

                // Activation function.
                hidden[i] = sigmoid(hidden[i]);
            }

            // For output layer.
            for (int i = 0; i < output_size; i++)
            {
                // Summation operation.
                for (int j = 0; j < hidden_size; j++)
                {
                    output[i] += hidden[j] * weights_hidden_output[j][i];
                }
                output[i] += bias_output[i];

                // Activation function.
                output[i] = sigmoid(output[i]);
            }

            // Backpropagation: 

            // Calculate the error from each output neuron.
            vector<double> output_error(output_size, 0.0);
            for (int i = 0; i < output_size; i++)
            {
                output_error[i] = target[i] - output[i];
            }

            // Send the error to the hidden layer.
            vector<double> hidden_error(hidden_size, 0.0);
            for (int i = 0; i < hidden_size; i++)
            {
                for (int j = 0; j < output_size; j++)
                {
                    hidden_error[i] += output_error[j] * weights_hidden_output[i][j];
                }
            }

            // RMSProp weight and bais update:

            // Hidden to Output weights.
            for (int i = 0; i < hidden_size; i++)
            {
                for (int j = 0; j < output_size; j++)
                {
                    // Calculate gradients.
                    double gradient = output_error[j] * hidden[i] * sigmoid_derivative(output[j]);

                    // Update the RMSProp cache.
                    // 0.9 is the gradient decay rate.
                    // 0.1 is the weight of the current squared gradient in the update, determining how quickly the chache variable updates.
                    rmsprop_cache_hidden_output[i][j] = 0.9 * rmsprop_cache_hidden_output[i][j] + 0.1 * gradient * gradient;

                    // Update weights with RMSProp.
                    // 1e-8 is the epsilon term, a constant added for stability.
                    weights_hidden_output[i][j] += (learning_rate / sqrt(rmsprop_cache_hidden_output[i][j] + 1e-8)) * gradient;
                }
            }

            // Input to Hidden weights.
            for (int i = 0; i < input_size; i++)
            {
                for (int j = 0; j < hidden_size; j++)
                {
                    // Calculate gradients.
                    gradient_weight = hidden_error[j] * input[i] * sigmoid_derivative(hidden[j]);

                    // Update the RMSProp cache.
                    rmsprop_cache_input_hidden[i][j] = 0.9 * rmsprop_cache_input_hidden[i][j] + 0.1 * gradient_weight * gradient_weight;

                    // Update weights with RMSProp.
                    weights_input_hidden[i][j] += (learning_rate / sqrt(rmsprop_cache_input_hidden[i][j] + 1e-8)) * gradient_weight;
                }
            }

            // Output layer bias.
            for (int i = 0; i < output_size; i++)
            {
                // Calculate gradients for biases.
                double gradient = output_error[i] * sigmoid_derivative(output[i]);

                // Update the RMSProp cache for biases.
                rmsprop_cache_bias_output[i] = 0.9 * rmsprop_cache_bias_output[i] + 0.1 * gradient * gradient;

                // Update biases with RMSProp.
                bias_output[i] += (learning_rate / sqrt(rmsprop_cache_bias_output[i] + 1e-8)) * gradient;
            }

            // Hidden layer bias.
            for (int i = 0; i < hidden_size; i++)
            {
                // Calculate gradients for biases.
                double gradient = hidden_error[i] * sigmoid_derivative(hidden[i]);

                // Update the RMSProp cache for biases.
                rmsprop_cache_bias_hidden[i] = 0.9 * rmsprop_cache_bias_hidden[i] + 0.1 * gradient * gradient;

                // Update biases with RMSProp.
                bias_hidden[i] += (learning_rate / sqrt(rmsprop_cache_bias_hidden[i] + 1e-8)) * gradient;
            }
        }

        // Print epoch number and gradient.
        cout << "Epoch [" << (epoch + 1) << "] - Gradient: " << gradient_weight << endl;
    }
}

// feedForward function to predict output on the testing set using the updated weights and bias.
vector<double> NeuralNetwork::feedForward(const vector<double> &input)
{
    // Input to hidden layer feed forward
    vector<double> hidden(hidden_size, 0.0);
    for (int i = 0; i < hidden_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            hidden[i] += input[j] * weights_input_hidden[j][i];
        }
        hidden[i] += bias_hidden[i];
        hidden[i] = sigmoid(hidden[i]);
    }

    // Hidden to output layer feed forward
    vector<double> output(output_size, 0.0);
    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < hidden_size; j++)
        {
            output[i] += hidden[j] * weights_hidden_output[j][i];
        }
        output[i] += bias_output[i];
        output[i] = sigmoid(output[i]);
    }

    // Returning final predictions (class probabilities)
    return output;
}