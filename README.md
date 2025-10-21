### Project Structure: 
1) main.cpp file: The file implementing the main function for model training, testing, and evaluation
2) NeuralNet.cpp file: Contains the implementation of Neural Network class and its functions declared in NeuralNet.h file.
3) Util.cpp file: Contains the implementation of utility functions declared in Utils.h file.
4) dataset.ipynb file: Jupyter notebook for simple dataset visualization.
5) evaluation.ipynb file: Jupyter notebook for result evaluation.
6) data folder: Folder containing datasets and results csv file.

### For running the model:
- Change the file path in the main.cpp file to select different datasets.
- Compile the project using the command:
    - g++ main.cpp NeuralNet.cpp Utils.cpp -o main
- Run the code using the command:
    - './main' on Windows PowerShell.
    - 'main.exe' on Windows Command Prompt.
- Run the evaluation.ipynb to plot ROC curve and Confusion matrix.