#include "Header.h"

using namespace std;

double learningRate = 0.01;
double* outputErrorGradients;
double** deltaHiddenOutput;

int main(void)
{
	//Write foward pass through one layer network to make sure it works

	//setup
	getInputData();
	initialiseNetwork();

	//-------------------------------------------------Check weights---------------------------------------
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			cout << "Input-Hidden (" << j << "," << i << ") wInputCell connection is: " << iHWeights[j][i].wInputCell << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputInputGate connection is: " << iHWeights[j][i].wInputInputGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputForgetGate connection is: " << iHWeights[j][i].wInputForgetGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputOutputGate connection is: " << iHWeights[j][i].wInputOutputGate << endl;
		}
		cout << "wCellIn for " << i << " is: " << hiddenLayer[i].wCellIn << endl;
		cout << "wCellForget for " << i << " is: " << hiddenLayer[i].wCellForget << endl;
		cout << "wCellOut for " << i << " is: " << hiddenLayer[i].wCellOut << endl;
	}

	//update weights for hidden to output layer
	for (int j = 0; j <= HIDDEN; j++)
	{
		for (int k = 0; k < OUTPUT; k++)
		{
			cout << "Hidden-Output (" << j << "," << k << ") connection is: " << hOWeights[j][k] << endl;			
		}
	}
	//-----------------------------------------------------------------------------------------------------

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

	//put variables for derivaties in weight class and cell class

	//partial derivatives for cell input
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			iHWeights[j][i].dSInputCell = iHWeights[j][i].dSInputCell*hiddenLayer[i].yForget + gPrime(hiddenLayer[i].netCellState)*hiddenLayer[i].yIn*inputLayer[j];
		}
	}

	//partial derivatives for input gate
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			iHWeights[j][i].dSInputInputGate = iHWeights[j][i].dSInputInputGate*hiddenLayer[i].yForget + activationFunctionG(hiddenLayer[i].netCellState)*fPrime(hiddenLayer[i].netIn)*inputLayer[j];
		}

		//partial derivatives for internal connections
		hiddenLayer[i].dSWCellIn = hiddenLayer[i].dSWCellIn*hiddenLayer[i].yForget + activationFunctionG(hiddenLayer[i].netCellState)*fPrime(hiddenLayer[i].netIn)*hiddenLayer[i].cellState;
	}

	//partial derivatives for forget gate
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			//initially this equals zero as the initial dS is zero and the previous cell state is zero
			iHWeights[j][i].dSInputForgetGate = iHWeights[j][i].dSInputForgetGate*hiddenLayer[i].yForget + hiddenLayer[i].previousCellState*fPrime(hiddenLayer[i].netForget)*inputLayer[j];
		}

		//partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
		hiddenLayer[i].dSWCellForget = hiddenLayer[i].dSWCellForget*hiddenLayer[i].yForget + hiddenLayer[i].previousCellState*fPrime(hiddenLayer[i].netForget)*hiddenLayer[i].previousCellState;
	}

	//backward pass

	//create gradient list
	outputErrorGradients = new(double[OUTPUT]);
	for (int i = 0; i < OUTPUT; i++) outputErrorGradients[i] = 0;

	//create delta list
	deltaHiddenOutput = new(double*[HIDDEN+1]);
	for (int i = 0; i <= HIDDEN; i++)
	{
		deltaHiddenOutput[i] = new(double[OUTPUT]);
		for (int j = 0; j < OUTPUT; j++) deltaHiddenOutput[i][j] = 0;
	}
	
	//for all output neurons
	for (int k = 0; k < OUTPUT; k++)
	{
		//output layer of linear neurons. find the difference between target and output
		outputErrorGradients[k] = (inputData[1] - outputLayer[k]);

		//for each connection to the hidden layer
		for (int j = 0; j <= HIDDEN; j++)
		{
			deltaHiddenOutput[j][k] += learningRate*hiddenLayer[j].cellOutput*outputErrorGradients[k];
		}
	}

	//for each hidden neuron
	for (int j = 0; j < HIDDEN; j++)
	{
		//find the error by find the product of the output errors and their weight connection.
		double weightedSum = 0;
		for (int k = 0; k < OUTPUT; k++)
		{
			weightedSum += outputErrorGradients[k]*hOWeights[j][k];
		}

		//using the error find the gradient of the output gate
		hiddenLayer[j].gradientOutputGate = fPrime(hiddenLayer[j].netOut)*hiddenLayer[j].cellState*weightedSum;

		//internal cell state error
		hiddenLayer[j].cellStateError = hiddenLayer[j].yOut*weightedSum;
	}
	
	//weight updates

	//already done the deltas for the hidden-output connections

	//output gates. for each connection to the hidden layer
	for (int i = 0; i < HIDDEN; i++)
	{
		//to the input layer
		for (int j = 0; j <= INPUT; j++)
		{
			//make the delta equal to the learning rate multiplied by the gradient multipled by the input for the connection
			iHWeights[j][i].deltaOutputGateInput = learningRate*hiddenLayer[i].gradientOutputGate*inputLayer[j];
		}

		//for the internal connection
		hiddenLayer[i].deltaOutputGateCell = learningRate*hiddenLayer[i].gradientOutputGate*hiddenLayer[i].cellState;
	}

	//input gates. for each connection from the hidden layer
	for (int i = 0; i < HIDDEN; i++)
	{
		//to the input layer
		for (int j = 0; j <= INPUT; j++)
		{
			//using partial derivative from input to input gate
			iHWeights[j][i].deltaInputGateInput = learningRate*hiddenLayer[i].cellStateError*iHWeights[j][i].dSInputInputGate;
		}

		//using internal partial derivative
		hiddenLayer[i].deltaInputGateCell = learningRate*hiddenLayer[i].cellStateError*hiddenLayer[i].dSWCellIn;
	}

	//forget gates. for each connection from the hidden layer
	for (int i = 0; i < HIDDEN; i++)
	{
		//to the input layer
		for (int j = 0; j <= INPUT; j++)
		{
			iHWeights[j][i].deltaForgetGateInput = learningRate*hiddenLayer[i].cellStateError*iHWeights[j][i].dSInputForgetGate;
		}
		hiddenLayer[i].deltaForgetGateCell = learningRate*hiddenLayer[i].cellStateError*hiddenLayer[i].dSWCellForget;
	}

	//cell inputs. for each connection from the hidden layer
	for (int i = 0; i < HIDDEN; i++)
	{
		//to the input layer
		for (int j = 0; j <= INPUT; j++)
		{
			iHWeights[j][i].deltaInputCellInput = learningRate*hiddenLayer[i].cellStateError*iHWeights[j][i].dSInputCell;
		}
	}

	//updates weights for input to hidden layer
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			//update connection weights
			iHWeights[j][i].wInputCell += iHWeights[j][i].deltaInputCellInput;
			iHWeights[j][i].wInputInputGate += iHWeights[j][i].deltaInputGateInput;
			iHWeights[j][i].wInputForgetGate += iHWeights[j][i].deltaForgetGateInput;
			iHWeights[j][i].wInputOutputGate += iHWeights[j][i].deltaOutputGateInput;
		}

		//update internal weights
		hiddenLayer[i].wCellIn += hiddenLayer[i].deltaInputGateCell;
		hiddenLayer[i].wCellForget += hiddenLayer[i].deltaForgetGateCell;
		hiddenLayer[i].wCellOut += hiddenLayer[i].deltaOutputGateCell;
	}

	//update weights for hidden to output layer
	for (int j = 0; j <= HIDDEN; j++)
	{
		for (int k = 0; k < OUTPUT; k++)
		{
			hOWeights[j][k] += deltaHiddenOutput[j][k];
		}
	}

	cout << endl << endl << "After weight update" << endl << endl;

	//-------------------------------------------------Check weights---------------------------------------
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			cout << "Input-Hidden (" << j << "," << i << ") wInputCell connection is: " << iHWeights[j][i].wInputCell << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputInputGate connection is: " << iHWeights[j][i].wInputInputGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputForgetGate connection is: " << iHWeights[j][i].wInputForgetGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputOutputGate connection is: " << iHWeights[j][i].wInputOutputGate << endl;
		}
		cout << "wCellIn for " << i << " is: " << hiddenLayer[i].wCellIn << endl;
		cout << "wCellForget for " << i << " is: " << hiddenLayer[i].wCellForget << endl;
		cout << "wCellOut for " << i << " is: " << hiddenLayer[i].wCellOut << endl;
	}

	//update weights for hidden to output layer
	for (int j = 0; j <= HIDDEN; j++)
	{
		for (int k = 0; k < OUTPUT; k++)
		{
			cout << "Hidden-Output (" << j << "," << k << ") connection is: " << hOWeights[j][k] << endl;			
		}
	}
	//-----------------------------------------------------------------------------------------------------



	cout << "Press any key to exit!";
	cin.get();
	return(0);
}

double activationFunctionG(double x)
{
	//sigmoid function return a bounded output between [-2,2]
	return (4/(1+exp(-x)))-2;
}

double gPrime(double x)
{
	return 4*activationFunctionF(x)*(1-activationFunctionF(x));
}

double activationFunctionF(double x)
{
	return (1/(1+exp(-x)));
}

double fPrime(double x)
{
	return activationFunctionF(x)*(1-activationFunctionF(x));
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


