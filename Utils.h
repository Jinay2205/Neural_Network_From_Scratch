// Utility Function and variable declarations.

#pragma once

#include <string>
#include <vector>

using namespace std;

// Defining a structure for storing the data read from the csv file.
struct CSVData
{
    vector<string> header;
    vector<vector<double>> data;
    vector<vector<double>> target;
};

// Defining a structure for storing split data.
struct TrainTest
{
    vector<vector<double>> train_data;
    vector<vector<double>> train_target;
    vector<vector<double>> test_data;
    vector<vector<double>> test_target;
};

// Defining a structure for storing evaluation metrics.
struct Metrics
{
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    int tp, tn, fp, fn;
};

// Sigmoid activation function.
double sigmoid(double x);

// Derivative of the sigmoid function.
double sigmoid_derivative(double x);

// Function to read data from csv file.
CSVData read_csv(string file_name);

// Function to split data into training and testing sets.
TrainTest train_test_split(vector<vector<double>> data, vector<vector<double>> target, double test_size);

// Function for min-max scaling the input data.
vector<vector<double>> scaler(const vector<vector<double>> data);

// Function to store prediction (classification) results in a csv file. 
void save_results(const vector<vector<double>> &actual, const vector<vector<double>> &predictions, const vector<int> &classifications, const string &filename);

// Function to calculate evaluation metrics.
Metrics calculate_metrics(const vector<vector<double>> &test_target, const vector<int> &classifications);