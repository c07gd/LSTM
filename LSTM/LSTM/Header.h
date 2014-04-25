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
int LENGTH;
double* inputData;
double* outputData;

//T3
double* inputLayerT3;
LSTMWeight** iHWeightsT3;
LSTMCell* hiddenLayerT3;
double** hOWeights; //Dont need now
double* outputLayer; // dont need now

//T2
double* inputLayerT2;
LSTMWeight** iHWeightsT2;
LSTMCell* hiddenLayerT2;
LSTMWeight** rHWeightsT2;

//T1
double* inputLayerT1;
LSTMWeight** iHWeightsT1;
LSTMCell* hiddenLayerT1;
LSTMWeight** rHWeightsT1;

//T0
double* inputLayerT0;
LSTMWeight** iHWeightsT0;
LSTMCell* hiddenLayerT0;
LSTMWeight** rHWeightsT0;


//other
double learningRate = 0.01;
double* outputErrorGradients;
double** deltaHiddenOutput;

//prototypes
void updateWeights(void);
void backwardPass(int index);
void hObackwardPass(int index);
void partialDerivatives(void);
void recurrentPartialDerivatives(double* iL, LSTMWeight** wIH, LSTMCell* rL, LSTMWeight** wRH, LSTMCell* hL);
void calculateOutputLayer(void);
void calculatehiddenLayer(void);
void calculateRecurrentLayer(double* iL, LSTMWeight** wIH, LSTMCell*rL, LSTMWeight** wRH, LSTMCell* hL);
void passInputData(double* iL, int index);
void checkWeights(void);
void initialiseNetwork(void);
void copyWeights(void);
void getOutputData(void);
void getInputData(void);
double activationFunctionF(double x);
double fPrime(double x);
double activationFunctionG(double x);
double gPrime(double x);