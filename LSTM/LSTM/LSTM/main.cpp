#include <stdio.h>
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>
#include "LSTM.h"

using namespace std;

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
double activationFunctionG(double x);

int main(void)
{
	//Write foward pass through one layer network to make sure it works

	//setup
	getInputData();
	initialiseNetwork();

	//IMPORTANT!!! Make sure when put in loop that previousCellState = cellState
	//pass data to input neuron
	for (int i = 0; i < INPUT; i++) inputLayer[i] = inputData[0];

	//loop through all input gates in hidden layer
	//for each hidden neuron
	for (int j = 0; j < HIDDEN; j++)
	{
		//rest the value of the net input to zero
		hiddenLayer[j].netIn = 0;

		//for each input neuron
		for (int i = 0; i <= INPUT; i++)
		{
			//multiply each input neuron by the connection to that hidden layer input gate
			hiddenLayer[j].netIn += inputLayer[i]*iHWeights[i][j].wInputInputGate;
		}

		//include internal connection multiplied by the previous cell state
		hiddenLayer[j].netIn += hiddenLayer[j].previousCellState*hiddenLayer[j].wCellIn;

		//squash input
		hiddenLayer[j].yIn = activationFunctionF(hiddenLayer[j].netIn);
	}

	//loop through all forget gates in hiddden layer
	for (int j = 0; j < HIDDEN; j++)
	{
		hiddenLayer[j].netForget = 0;
		for (int i = 0; i <= INPUT; i++)
		{
			hiddenLayer[j].netForget += inputLayer[i]*iHWeights[i][j].wInputForgetGate;
		}
		//include internal connection multiplied by the previous cell state
		hiddenLayer[j].netForget += hiddenLayer[j].previousCellState*hiddenLayer[j].wCellForget;
		hiddenLayer[j].yForget = activationFunctionF(hiddenLayer[j].netForget);
	}

	//loop through all cell inputs in hidden layer
	for (int j = 0; j < HIDDEN; j++)
	{
		//reset each netCell state to zero
		hiddenLayer[j].netCellState = 0;

		//loop through all connection to input layer
		for (int i = 0; i <= INPUT; i++)
		{
			hiddenLayer[j].netCellState += inputLayer[i]*iHWeights[i][j].wInputCell;
		}

		//cell state is equal to the previous cell state multipled by the forget gate and the cell inputs multiplied by the input gate
		hiddenLayer[j].cellState = hiddenLayer[j].yForget*hiddenLayer[j].previousCellState + hiddenLayer[j].yIn*activationFunctionG(hiddenLayer[j].netCellState);
	}

	//loop through all output gate in hidden layer
	for (int j = 0; j < HIDDEN; j++)
	{
		//reset each netOut to zero
		hiddenLayer[j].netOut = 0;

		//For each input
		for (int i = 0; i <= INPUT; i++)
		{
			//multiply the input with the connection to that input
			hiddenLayer[j].netOut += inputLayer[i]*iHWeights[i][j].wInputOutputGate;
		}

		//include the internal connection multiplied by the CURRENT cell state
		hiddenLayer[j].netOut += hiddenLayer[j].cellState*hiddenLayer[j].wCellOut;

		//squash output gate 
		hiddenLayer[j].yOut = activationFunctionF(hiddenLayer[j].netOut);
	}

	for (int j = 0; j < HIDDEN; j++)
	{
		hiddenLayer[j].cellOutput = hiddenLayer[j].cellState*hiddenLayer[j].yOut;
	}

	for (int k = 0; k < OUTPUT; k++)
	{
		outputLayer[k] = 0;
		for (int j = 0; j <= HIDDEN; j++)
		{
			outputLayer[k] += hiddenLayer[j].cellOutput*hOWeights[j][k];
		}
	}

	for (int k = 0; k < OUTPUT; k++)
	{
		cout << "Output is: " << outputLayer[k] << endl;
	}

	

	//for (int j = 0; j < HIDDEN; j++)
	//{
	//	cout << "Value of netCellState in LSTMCell " << j << " is: " << hiddenLayer[j].netCellState << endl;
	//	cout << "Value of cellOutput in LSTMCell " << j << " is: " << hiddenLayer[j].cellOutput << endl << endl;
	//}



	cout << "Press any key to exit!";
	cin.get();
	return(0);
}

double activationFunctionG(double x)
{
	//sigmoid function return a bounded output between [-2,2]
	return (4/(1+exp(-x)))-2;
}

double activationFunctionF(double x)
{
	return (1/(1+exp(-x)));
}

void getInputData(void)
{
	//417483

	inputData = new(double[MAX_LENGTH]);	
	char data[10];
	string fileName = "example2.csv";
	ifstream infile;
	infile.open(fileName);
	if(infile.is_open())
	{
		cout << fileName << " opened successfully!!!. Writing data from array to file" << endl;
		for(int row = 0; (row < MAX_LENGTH) && (!infile.eof()); row++)
		{
			infile.getline(data,10,',');
			inputData[row] = atof(data);
			LENGTH++;
		}
	}
	infile.close();
	cout << "Array size: " << LENGTH << endl;
}


void initialiseNetwork(void)
{
	//create and initalise input and bias neuron.
	inputLayer = new(double[INPUT+1]);
	for (int i = 0; i < INPUT; i++) inputLayer[i] = 0;
	inputLayer[INPUT] = -1;

	//create and initialise the weights from input to hidden layer
	iHWeights = new(LSTMWeight*[INPUT+1]);
	for (int i = 0; i <= INPUT; i++)
	{
		iHWeights[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			iHWeights[i][j].initialise(INPUT);
		}
	}

	//Create LSTM hidden layer
	hiddenLayer = new(LSTMCell[HIDDEN+1]);
	for (int i = 0; i < HIDDEN; i++) hiddenLayer[i].initialise(NORMAL);
	hiddenLayer[HIDDEN].initialise(BIAS);

	//Create and intialise the weights from hidden to output layer, these are just normal weights
	double hiddenOutputRand = 1/sqrt(double(HIDDEN));
	hOWeights = new(double*[HIDDEN+1]);
	for (int i = 0; i <= HIDDEN; i++)
	{
		hOWeights[i] = new(double[OUTPUT]);
		for (int j = 0; j < OUTPUT; j++)
		{
			hOWeights[i][j] = (((double)((rand()%100)+1)/100)*2*hiddenOutputRand)-hiddenOutputRand;
		}
	}

	//create output layer
	outputLayer = new(double[OUTPUT]);
	for (int i = 0; i < OUTPUT; i++) outputLayer[i] = 0;
}


