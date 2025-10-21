#include "Utils.h"
#include "NeuralNet.h"
#include <iostream>

using namespace std;

int main()
{
    string data_file = "./data/breast_cancer.csv";

    // read the CSV file
    CSVData csv_data = read_csv(data_file);
    vector<string> header = csv_data.header;
    vector<vector<double>> data = csv_data.data;
    vector<vector<double>> target = csv_data.target;

    vector<vector<double>> predictions;
    vector<int> classifications;

    // Scaling the input data
    vector<vector<double>> scaled_data = scaler(data);

    // train_test_split
    TrainTest split_data = train_test_split(scaled_data, target, 0.3);

    vector<vector<double>> train_data = split_data.train_data;
    vector<vector<double>> train_target = split_data.train_target;
    vector<vector<double>> test_data = split_data.test_data;
    vector<vector<double>> test_target = split_data.test_target;

    // Define the network architecture
    int input_size = train_data[0].size();
    int hidden_size = 32;
    int output_size = 1;
    double threshold = 0.5;

    // Create a neural network
    NeuralNetwork nn(input_size, hidden_size, output_size);

    cout << "Input Size: " << train_data[0].size() << endl;

    // Train the network
    double learning_rate = 0.1;
    int epochs = 3;

    cout << "Enter learning rate: ";
    cin >> learning_rate;

    cout << "Enter number of epochs: ";
    cin >> epochs;

    nn.train(train_data, train_target, learning_rate, epochs);
    cout << "\nTraining complete.\n"
         << endl;

    for (size_t i = 0; i < test_data.size(); i++)
    {
        vector<double> input = test_data[i];
        vector<double> prediction = nn.feedForward(input);
        predictions.push_back(prediction);

        double binary_prediction = (prediction[0] >= threshold) ? 1 : 0;

        classifications.push_back(binary_prediction);
    }

    // Printing Actual vs. Pedicted values
    // cout << "Actual vs. Predicted:" << endl;
    // for (size_t i = 0; i < test_data.size(); i++)
    // {
    //     int actual = static_cast<int>(test_target[i][0]);
    //     int classification = static_cast<int>(classifications[i]);

    //     cout << "Actual: " << actual << " | Predicted: " << classification << endl;
    // }

    // Calculating evaluation metrics: Accuracy, Precision, Recall
    Metrics metrics = calculate_metrics(test_target, classifications);

    cout << "Testing Evaluation Metrics: " << endl;
    cout << "Accuracy  : " << metrics.accuracy << endl;
    cout << "Precision : " << metrics.precision << endl;
    cout << "Recall    : " << metrics.recall << endl;
    cout << "F1-Score  : " << metrics.f1_score << endl;
    cout << "TP        : " << metrics.tp << endl;
    cout << "TN        : " << metrics.tn << endl;
    cout << "FP        : " << metrics.fp << endl;
    cout << "FN        : " << metrics.fn << endl;

    save_results(test_target, predictions, classifications, "./data/results.csv");

    return 0;
}
