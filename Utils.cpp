// Implementation of the utility functions.

// Importing header file containing declarations.
#include "Utils.h"

// Importing libraries.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

using namespace std;

// Sigmoid activation function.
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Sigmoid derivative for use in optimization.
double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

// Function to read data from CSV files. It returns an object of type Struct CSVData.
CSVData read_csv(string file_name)
{
    ifstream file(file_name);
    // Initializing the object
    CSVData csv_data;

    if (!file)
    {
        cerr << "Error opening the file." << endl;
        return csv_data;
    }

    // Initializing separate vectors for header row, input data, and the target column.
    vector<string> header;
    vector<vector<double>> data;
    vector<vector<double>> target;

    string line;

    // Reading the header row and pushing it into the header vector.
    if (getline(file, line))
    {
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ','))
        {
            csv_data.header.push_back(cell);
        }
    }

    // Reading subsequent rows and populating the data and target vectors.
    while (getline(file, line))
    {
        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ','))
        {
            double value = stod(cell);
            row.push_back(value);
        }

        // Assuming the last column (last value of a row vector) to be the target.
        vector<double> target_val = {row.back()};
        csv_data.target.push_back(target_val);

        // Removing the target value from the row vector.
        row.pop_back();
        csv_data.data.push_back(row);
    }

    // Close the file.
    file.close();
    return csv_data;
}

// Function to split train and test sets.
TrainTest train_test_split(vector<vector<double>> data, vector<vector<double>> target, double test_size)
{
    TrainTest split_data;

    if (data.size() != target.size() || data.empty() || target.empty())
    {
        cerr << "Invalid data or target vectors." << endl;
        return split_data;
    }

    // Initializing size variables.
    size_t total_samples = data.size();
    size_t test_samples = static_cast<size_t>(total_samples * test_size);
    size_t train_samples = total_samples - test_samples;

    // Updating shuffled indices.
    vector<size_t> indices(total_samples);
    for (size_t i = 0; i < total_samples; i++)
    {
        indices[i] = i;
    }

    // Shuffling the data using a Pseudo-Random Number Generator (mt19937).
    shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

    // Split the data and target into training and testing sets.
    split_data.train_data.resize(train_samples);
    split_data.train_target.resize(train_samples);
    split_data.test_data.resize(test_samples);
    split_data.test_target.resize(test_samples);

    // Populating the data and target vectors.
    for (size_t i = 0; i < total_samples; i++)
    {
        size_t index = indices[i];
        if (i < train_samples)
        {
            split_data.train_data[i] = data[index];
            split_data.train_target[i] = target[index];
        }
        else
        {
            split_data.test_data[i - train_samples] = data[index];
            split_data.test_target[i - train_samples] = target[index];
        }
    }

    return split_data;
}

// Function for min-max scaling.
vector<vector<double>> scaler(const vector<vector<double>> data)
{
    // Create a copy of the original data.
    vector<vector<double>> scaled_data = data;

    // Check if the data is empty.
    if (data.empty() || data[0].empty())
    {
        cerr << "Input data is empty." << endl;
        return scaled_data;
    }

    int num_columns = data[0].size();

    // Initializing minimum and maximum values of each column.
    vector<double> min_values(num_columns, numeric_limits<double>::max());    // Returns the max value of the double data type dependent on the compiler.
    vector<double> max_values(num_columns, numeric_limits<double>::lowest()); // Similarly returns the minimum value of the double data type.

    // Updating the minimum and maximum values by looping through each value.
    for (const vector<double> &row : data)
    {
        for (int col = 0; col < num_columns; col++)
        {
            min_values[col] = min(min_values[col], row[col]);
            max_values[col] = max(max_values[col], row[col]);
        }
    }

    // Scaling the data.
    for (vector<double> &row : scaled_data)
    {
        for (int col = 0; col < num_columns; col++)
        {
            // If min and max values are different.
            if (max_values[col] - min_values[col] != 0)
            {
                // Perform scaling.
                row[col] = (row[col] - min_values[col]) / (max_values[col] - min_values[col]);
            }
            // If min and max are same.
            else
            {
                // Avoid division by zero.
                row[col] = 0.0;
            }
        }
    }

    return scaled_data;
}

// Function to save results in a separate csv file.
void save_results(const vector<vector<double>> &actual, const vector<vector<double>> &predictions, const vector<int> &classifications, const string &filename)
{
    // Create a file to store the results.
    ofstream resultFile(filename);

    if (!resultFile.is_open())
    {
        cerr << "Error opening the result file." << endl;
        return;
    }

    // Write the header to the result file.
    resultFile << "Actual,Prediction,Classification" << endl;

    // Loop through the data and write the actual, prediction, and classification to the file.
    for (size_t i = 0; i < actual.size(); i++)
    {
        resultFile << actual[i][0] << "," << predictions[i][0] << "," << classifications[i] << endl;
    }

    // Close the result file.
    resultFile.close();

    cout << "\nResults saved in '" << filename << "'" << endl;
}

// Function to calculate evaluation metrics.
Metrics calculate_metrics(const vector<vector<double>> &test_target, const vector<int> &classifications)
{
    Metrics metrics;

    // Casting the 2D target vector into a single row vector.
    vector<int> target;
    target.reserve(test_target.size());

    // Extracting each row vector from the target column.
    for (const vector<double> &inner : test_target)
    {
        if (!inner.empty())
        {
            // Extracting the value and pushing it into the target vector.
            int value = static_cast<int>(inner[0]);
            target.push_back(value);
        }
        else
        {
            cerr << "Empty inner vector found." << endl;
        }
    }

    // Checking for size mismatch or if vectors are empty.
    if (target.size() != classifications.size() || target.empty() || classifications.empty())
    {
        cerr << "Invalid input vectors." << endl;
        return metrics;
    }

    // Initialize the TP, TN, FP and FN values.
    int true_positives = 0;
    int true_negatives = 0;
    int false_positives = 0;
    int false_negatives = 0;

    size_t total_samples = target.size();
    // Iterate through the total number of samples in the testing set.
    for (size_t i = 0; i < total_samples; i++)
    {
        if (target[i] == 1)
        {
            if (classifications[i] == 1)
            {
                true_positives++; // If target = 1, classification = 1.
            }
            else
            {
                false_positives++; // If target = 1, classification = 0.
            }
        }
        else
        {
            if (classifications[i] == 0)
            {
                true_negatives++; // If target = 0, classification = 0.
            }
            else
            {
                false_negatives++; // If target = 0, classification = 1.
            }
        }
    }

    // Calculating accuracy, precision, recall, f1_score and assigning values.
    metrics.accuracy = static_cast<double>(true_positives + true_negatives) / total_samples;
    metrics.precision = static_cast<double>(true_positives) / (true_positives + false_positives);
    metrics.recall = static_cast<double>(true_positives) / (true_positives + false_negatives);
    metrics.f1_score = static_cast<double>(2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
    metrics.tp = true_positives, metrics.fp = false_positives, metrics.tn = true_negatives, metrics.fn = false_negatives;

    return metrics;
}
