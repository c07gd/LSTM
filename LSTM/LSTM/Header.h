#include <stdio.h>
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>
#include "LSTM.h"

//Constants
const int INPUT = 1;
const int HIDDEN = 10;
const int OUTPUT = 1;

//for LSTM layer
const bool NORMAL = true;
const bool BIAS = false;

//data input
const int MAX_LENGTH = 500000;
int LENGTH = 0;
double* inputData;

//Layers
double* inputLayer;
LSTMWeight** iHWeights;
LSTMCell* hiddenLayer;
double** hOWeights;
double* outputLayer;

//prototypes
void initialiseNetwork(void);
void getInputData(void);
double activationFunctionF(double x);
double fPrime(double x);
double activationFunctionG(double x);
double gPrime(double x);