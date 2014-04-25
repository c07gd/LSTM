#include "Header.h"

using namespace std;

int main(void)
{
	//Write foward pass through one layer network to make sure it works

	//setup
	getInputData();
	getOutputData();
	initialiseNetwork();

	double meanSquareError;

	for (int j = 0; j < 1; j++)
	{
		meanSquareError = 0;
		//for (int i = 3; i < (LENGTH-2); i++)
		for (int i = 3; i < 4; i++)
		{
			//make all weights the same and therefore layers identical
			copyWeights();
			checkWeights();

			//calculate T3 layer
			passInputData(inputLayerT3, i-3);
			calculatehiddenLayer();
			partialDerivatives();

			//calculate T2 layer
			passInputData(inputLayerT2, i-2);
			calculateRecurrentLayer(inputLayerT2,iHWeightsT2, hiddenLayerT3, rHWeightsT2, hiddenLayerT2);
			recurrentPartialDerivatives(inputLayerT2,iHWeightsT2, hiddenLayerT3, rHWeightsT2, hiddenLayerT2);

			//calculate T1 Layer
			passInputData(inputLayerT1, i-1);
			calculateRecurrentLayer(inputLayerT1,iHWeightsT1, hiddenLayerT2, rHWeightsT1, hiddenLayerT1);
			recurrentPartialDerivatives(inputLayerT1,iHWeightsT1, hiddenLayerT2, rHWeightsT1, hiddenLayerT1);

			//calculate T0 Layer
			passInputData(inputLayerT0, i-0);
			calculateRecurrentLayer(inputLayerT0,iHWeightsT0, hiddenLayerT1, rHWeightsT0, hiddenLayerT0);
			recurrentPartialDerivatives(inputLayerT0,iHWeightsT0, hiddenLayerT1, rHWeightsT0, hiddenLayerT0);
			
			//output layer
			calculateOutputLayer();

			//backward pass
			hObackwardPass(i);


			cout << outputLayer[0] << endl;

			
			
			//backwardPass(i);
			//updateWeights();
			//meanSquareError += (powf(outputData[i] - outputLayer[0], 2))/LENGTH;
		}
		//cout << meanSquareError << "%" << endl;
	}

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
	LENGTH = 0;
	inputData = new(double[MAX_LENGTH]);	
	char data[10];
	/*string fileName = "example2.csv";*/
	string fileName = "testIn.csv";
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
	cout << "Input array size: " << LENGTH << endl;
}

void getOutputData(void)
{
	LENGTH = 0;
	outputData = new(double[MAX_LENGTH]);	
	char data[10];
	/*string fileName = "example2.csv";*/
	string fileName = "testOut.csv";
	ifstream infile;
	infile.open(fileName);
	if(infile.is_open())
	{
		cout << fileName << " opened successfully!!!. Writing data from array to file" << endl;
		for(int row = 0; (row < MAX_LENGTH) && (!infile.eof()); row++)
		{
			infile.getline(data,10,',');
			outputData[row] = atof(data);
			LENGTH++;
		}
	}
	infile.close();
	cout << "Output array size: " << LENGTH << endl;
}

void initialiseNetwork(void)
{
	//create and initalise input and bias neuron.
	inputLayerT3 = new(double[INPUT+1]);
	for (int i = 0; i < INPUT; i++) inputLayerT3[i] = 0;
	inputLayerT3[INPUT] = -1;

	//create and initialise the weights from input to hidden layer
	iHWeightsT3 = new(LSTMWeight*[INPUT+1]);
	for (int i = 0; i <= INPUT; i++)
	{
		iHWeightsT3[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			iHWeightsT3[i][j].initialise(INPUT);
		}
	}

	//Create LSTM hidden layer
	hiddenLayerT3 = new(LSTMCell[HIDDEN+1]);
	for (int i = 0; i < HIDDEN; i++) hiddenLayerT3[i].initialise(NORMAL);
	hiddenLayerT3[HIDDEN].initialise(BIAS);


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

	//-------------------------------------T2 Layer ---------------------------------

	//input T2 Layer
	inputLayerT2 = new(double[INPUT+1]);
	for (int i = 0; i < INPUT; i++) inputLayerT2[i] = 0;
	inputLayerT2[INPUT] = -1;

	//input T2 Weights
	iHWeightsT2 = new(LSTMWeight*[INPUT+1]);
	for (int i = 0; i <= INPUT; i++)
	{
		iHWeightsT2[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			iHWeightsT2[i][j].initialise(INPUT);
		}
	}

	//LSTM T2 Layer
	hiddenLayerT2 = new(LSTMCell[HIDDEN+1]);
	for (int i = 0; i < HIDDEN; i++) hiddenLayerT2[i].initialise(NORMAL);
	hiddenLayerT2[HIDDEN].initialise(BIAS);

	//LSTM T2 Weights
	rHWeightsT2 = new(LSTMWeight*[HIDDEN+1]);
	for (int i = 0; i <= HIDDEN; i++)
	{
		rHWeightsT2[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			rHWeightsT2[i][j].initialise(HIDDEN);
		}
	}

	//-------------------------------------T1 Layer ---------------------------------

	//input T1 Layer
	inputLayerT1 = new(double[INPUT+1]);
	for (int i = 0; i < INPUT; i++) inputLayerT1[i] = 0;
	inputLayerT1[INPUT] = -1;

	//input T1 Weights
	iHWeightsT1 = new(LSTMWeight*[INPUT+1]);
	for (int i = 0; i <= INPUT; i++)
	{
		iHWeightsT1[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			iHWeightsT1[i][j].initialise(INPUT);
		}
	}

	//LSTM T1 Layer
	hiddenLayerT1 = new(LSTMCell[HIDDEN+1]);
	for (int i = 0; i < HIDDEN; i++) hiddenLayerT1[i].initialise(NORMAL);
	hiddenLayerT1[HIDDEN].initialise(BIAS);

	//LSTM T1 Weights
	rHWeightsT1 = new(LSTMWeight*[HIDDEN+1]);
	for (int i = 0; i <= HIDDEN; i++)
	{
		rHWeightsT1[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			rHWeightsT1[i][j].initialise(HIDDEN);
		}
	}

	//-------------------------------------T0 Layer ---------------------------------

	//input T0 Layer
	inputLayerT0 = new(double[INPUT+1]);
	for (int i = 0; i < INPUT; i++) inputLayerT0[i] = 0;
	inputLayerT0[INPUT] = -1;

	//input T0 Weights
	iHWeightsT0 = new(LSTMWeight*[INPUT+1]);
	for (int i = 0; i <= INPUT; i++)
	{
		iHWeightsT0[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			iHWeightsT0[i][j].initialise(INPUT);
		}
	}

	//LSTM T0 Layer
	hiddenLayerT0 = new(LSTMCell[HIDDEN+1]);
	for (int i = 0; i < HIDDEN; i++) hiddenLayerT0[i].initialise(NORMAL);
	hiddenLayerT0[HIDDEN].initialise(BIAS);

	//LSTM T0 Weights
	rHWeightsT0 = new(LSTMWeight*[HIDDEN+1]);
	for (int i = 0; i <= HIDDEN; i++)
	{
		rHWeightsT0[i] = new(LSTMWeight[HIDDEN]);
		for (int j = 0; j < HIDDEN; j++)
		{
			rHWeightsT0[i][j].initialise(HIDDEN);
		}
	}

}

//makes all weights the same as the zeroth layer
void copyWeights(void)
{
	for(int i = 0; i <= INPUT; i++)
	{
		for (int j = 0; j < HIDDEN; j++)
		{
			//wInputCell same for all input layers
			iHWeightsT1[i][j].wInputCell = iHWeightsT0[i][j].wInputCell;
			iHWeightsT2[i][j].wInputCell = iHWeightsT0[i][j].wInputCell;
			iHWeightsT3[i][j].wInputCell = iHWeightsT0[i][j].wInputCell;

			//wInputInputGate same for all input layers
			iHWeightsT1[i][j].wInputInputGate = iHWeightsT0[i][j].wInputInputGate;
			iHWeightsT2[i][j].wInputInputGate = iHWeightsT0[i][j].wInputInputGate;
			iHWeightsT3[i][j].wInputInputGate = iHWeightsT0[i][j].wInputInputGate;

			//wInputForgetGate same for all input layers
			iHWeightsT1[i][j].wInputForgetGate = iHWeightsT0[i][j].wInputForgetGate;
			iHWeightsT2[i][j].wInputForgetGate = iHWeightsT0[i][j].wInputForgetGate;
			iHWeightsT3[i][j].wInputForgetGate = iHWeightsT0[i][j].wInputForgetGate;

			//wInputOutputGate same for all input layers
			iHWeightsT1[i][j].wInputOutputGate = iHWeightsT0[i][j].wInputOutputGate;
			iHWeightsT2[i][j].wInputOutputGate = iHWeightsT0[i][j].wInputOutputGate;
			iHWeightsT3[i][j].wInputOutputGate = iHWeightsT0[i][j].wInputOutputGate;
		}
	}

	for (int i = 0; i <= HIDDEN; i++)
	{
		for (int j = 0; j < HIDDEN; j++)
		{
			//wInputCell same for all input layers
			rHWeightsT1[i][j].wInputCell = rHWeightsT0[i][j].wInputCell;
			rHWeightsT2[i][j].wInputCell = rHWeightsT0[i][j].wInputCell;

			//wInputInputGate same for all input layers
			rHWeightsT1[i][j].wInputInputGate = rHWeightsT0[i][j].wInputInputGate;
			rHWeightsT2[i][j].wInputInputGate = rHWeightsT0[i][j].wInputInputGate;

			//wInputForgetGate same for all input layers
			rHWeightsT1[i][j].wInputForgetGate = rHWeightsT0[i][j].wInputForgetGate;
			rHWeightsT2[i][j].wInputForgetGate = rHWeightsT0[i][j].wInputForgetGate;

			//wInputOutputGate same for all input layers
			rHWeightsT1[i][j].wInputOutputGate = rHWeightsT0[i][j].wInputOutputGate;
			rHWeightsT2[i][j].wInputOutputGate = rHWeightsT0[i][j].wInputOutputGate;
		}

		//internal connections
		hiddenLayerT1[i].wCellIn = hiddenLayerT0[i].wCellIn;
		hiddenLayerT2[i].wCellIn = hiddenLayerT0[i].wCellIn;
		hiddenLayerT3[i].wCellIn = hiddenLayerT0[i].wCellIn;

		hiddenLayerT1[i].wCellForget = hiddenLayerT0[i].wCellForget;
		hiddenLayerT2[i].wCellForget = hiddenLayerT0[i].wCellForget;
		hiddenLayerT3[i].wCellForget = hiddenLayerT0[i].wCellForget;

		hiddenLayerT1[i].wCellOut = hiddenLayerT0[i].wCellOut;
		hiddenLayerT2[i].wCellOut = hiddenLayerT0[i].wCellOut;
		hiddenLayerT3[i].wCellOut = hiddenLayerT0[i].wCellOut;
	}
}

void checkWeights(void)
{
	//-------------------------------------------------Check weights---------------------------------------
	cout << "T3 Layer" << endl << endl;
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			cout << "Input-Hidden (" << j << "," << i << ") wInputCell connection is: " << iHWeightsT3[j][i].wInputCell << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputInputGate connection is: " << iHWeightsT3[j][i].wInputInputGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputForgetGate connection is: " << iHWeightsT3[j][i].wInputForgetGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputOutputGate connection is: " << iHWeightsT3[j][i].wInputOutputGate << endl;
		}
		cout << "wCellIn for " << i << " is: " << hiddenLayerT3[i].wCellIn << endl;
		cout << "wCellForget for " << i << " is: " << hiddenLayerT3[i].wCellForget << endl;
		cout << "wCellOut for " << i << " is: " << hiddenLayerT3[i].wCellOut << endl;
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
	cout << "T0 Layer" << endl << endl;
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			cout << "Input-Hidden (" << j << "," << i << ") wInputCell connection is: " << iHWeightsT0[j][i].wInputCell << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputInputGate connection is: " << iHWeightsT0[j][i].wInputInputGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputForgetGate connection is: " << iHWeightsT0[j][i].wInputForgetGate << endl;
			cout << "Input-Hidden (" << j << "," << i << ") wInputOutputGate connection is: " << iHWeightsT0[j][i].wInputOutputGate << endl;
		}
		cout << "wCellIn for " << i << " is: " << hiddenLayerT0[i].wCellIn << endl;
		cout << "wCellForget for " << i << " is: " << hiddenLayerT0[i].wCellForget << endl;
		cout << "wCellOut for " << i << " is: " << hiddenLayerT0[i].wCellOut << endl;
	}

	//update weights for hidden to output layer
	for (int j = 0; j <= HIDDEN; j++)
	{
		for (int k = 0; k < OUTPUT; k++)
		{
			cout << "Hidden-Output (" << j << "," << k << ") connection is: " << hOWeights[j][k] << endl;			
		}
	}


}

void passInputData(double* iL, int index)
{
	//pass data to input neuron
	for (int i = 0; i < INPUT; i++) iL[i] = inputData[index];
}

void calculatehiddenLayer(void)
{
	//calculate the hidden layer
	for (int j = 0; j < HIDDEN; j++)
	{
		//roll over cell state
		hiddenLayerT3[j].previousCellState = hiddenLayerT3[j].cellState;

		//rest the value of  to zero
		hiddenLayerT3[j].netIn = 0;
		hiddenLayerT3[j].netForget = 0;
		hiddenLayerT3[j].netCellState = 0;
		hiddenLayerT3[j].netOut = 0;

		//for each input neuron
		for (int i = 0; i <= INPUT; i++)
		{
			//multiply each input neuron by the connection to that hidden layer input gate
			hiddenLayerT3[j].netIn += inputLayerT3[i]*iHWeightsT3[i][j].wInputInputGate;
			hiddenLayerT3[j].netForget += inputLayerT3[i]*iHWeightsT3[i][j].wInputForgetGate;
			hiddenLayerT3[j].netCellState += inputLayerT3[i]*iHWeightsT3[i][j].wInputCell;
			hiddenLayerT3[j].netOut += inputLayerT3[i]*iHWeightsT3[i][j].wInputOutputGate;
		}

		//----------------------------recurrent layer-----------------------------
		//for (int i = 0; i <= HIDDEN; i++)
		//{
		//	hiddenLayerT3[j].netIn += recurrentLayer[i].cellOutput*rHWeights[i][j].wInputInputGate;
		//	hiddenLayerT3[j].netForget += recurrentLayer[i].cellOutput*rHWeights[i][j].wInputForgetGate;
		//	hiddenLayerT3[j].netCellState += recurrentLayer[i].cellOutput*rHWeights[i][j].wInputCell;
		//	hiddenLayerT3[j].netOut += recurrentLayer[i].cellOutput*rHWeights[i][j].wInputOutputGate;
		//}
		//------------------------------------------------------------------------

		//include internal connection multiplied by the previous cell state
		hiddenLayerT3[j].netIn += hiddenLayerT3[j].previousCellState*hiddenLayerT3[j].wCellIn;
		hiddenLayerT3[j].yIn = activationFunctionF(hiddenLayerT3[j].netIn);

		//forget gate
		hiddenLayerT3[j].netForget += hiddenLayerT3[j].previousCellState*hiddenLayerT3[j].wCellForget;
		hiddenLayerT3[j].yForget = activationFunctionF(hiddenLayerT3[j].netForget);

		//cell input
		hiddenLayerT3[j].cellState = hiddenLayerT3[j].yForget*hiddenLayerT3[j].previousCellState + hiddenLayerT3[j].yIn*activationFunctionG(hiddenLayerT3[j].netCellState);

		//output gate
		hiddenLayerT3[j].netOut += hiddenLayerT3[j].cellState*hiddenLayerT3[j].wCellOut;
		hiddenLayerT3[j].yOut = activationFunctionF(hiddenLayerT3[j].netOut);

		//cell output
		hiddenLayerT3[j].cellOutput = hiddenLayerT3[j].cellState*hiddenLayerT3[j].yOut;
	}
}

void calculateRecurrentLayer(double* iL, LSTMWeight** wIH, LSTMCell*rL, LSTMWeight** wRH, LSTMCell* hL)
{
	//calculate the hidden layer
	for (int j = 0; j < HIDDEN; j++)
	{
		//roll over cell state
		hL[j].previousCellState = hL[j].cellState;

		//rest the value of  to zero
		hL[j].netIn = 0;
		hL[j].netForget = 0;
		hL[j].netCellState = 0;
		hL[j].netOut = 0;

		//for each input neuron
		for (int i = 0; i <= INPUT; i++)
		{
			//multiply each input neuron by the connection to that hidden layer input gate
			hL[j].netIn += iL[i]*wIH[i][j].wInputInputGate;
			hL[j].netForget += iL[i]*wIH[i][j].wInputForgetGate;
			hL[j].netCellState += iL[i]*wIH[i][j].wInputCell;
			hL[j].netOut += iL[i]*wIH[i][j].wInputOutputGate;
		}

		//----------------------------recurrent layer-----------------------------
		for (int i = 0; i <= HIDDEN; i++)
		{
			hL[j].netIn += rL[i].cellOutput*wRH[i][j].wInputInputGate;
			hL[j].netForget += rL[i].cellOutput*wRH[i][j].wInputForgetGate;
			hL[j].netCellState += rL[i].cellOutput*wRH[i][j].wInputCell;
			hL[j].netOut += rL[i].cellOutput*wRH[i][j].wInputOutputGate;
		}
		//------------------------------------------------------------------------

		//include internal connection multiplied by the previous cell state
		hL[j].netIn += hL[j].previousCellState*hL[j].wCellIn;
		hL[j].yIn = activationFunctionF(hL[j].netIn);

		//forget gate
		hL[j].netForget += hL[j].previousCellState*hL[j].wCellForget;
		hL[j].yForget = activationFunctionF(hL[j].netForget);

		//cell input
		hL[j].cellState = hL[j].yForget*hL[j].previousCellState + hL[j].yIn*activationFunctionG(hL[j].netCellState);

		//output gate
		hL[j].netOut += hL[j].cellState*hL[j].wCellOut;
		hL[j].yOut = activationFunctionF(hL[j].netOut);

		//cell output
		hL[j].cellOutput = hL[j].cellState*hL[j].yOut;
	}
}

void calculateOutputLayer(void)
{
	//calulate output layer
	for (int k = 0; k < OUTPUT; k++)
	{
		outputLayer[k] = 0;
		for (int j = 0; j <= HIDDEN; j++)
		{
			outputLayer[k] += hiddenLayerT0[j].cellOutput*hOWeights[j][k];
		}
	}
}

void partialDerivatives(void)
{
	//partial derivatives for cell input
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			iHWeightsT3[j][i].dSInputCell = iHWeightsT3[j][i].dSInputCell*hiddenLayerT3[i].yForget + gPrime(hiddenLayerT3[i].netCellState)*hiddenLayerT3[i].yIn*inputLayerT3[j];
			iHWeightsT3[j][i].dSInputInputGate = iHWeightsT3[j][i].dSInputInputGate*hiddenLayerT3[i].yForget + activationFunctionG(hiddenLayerT3[i].netCellState)*fPrime(hiddenLayerT3[i].netIn)*inputLayerT3[j];
			//initially this equals zero as the initial dS is zero and the previous cell state is zero
			iHWeightsT3[j][i].dSInputForgetGate = iHWeightsT3[j][i].dSInputForgetGate*hiddenLayerT3[i].yForget + hiddenLayerT3[i].previousCellState*fPrime(hiddenLayerT3[i].netForget)*inputLayerT3[j];
		}

		//internal connection
		hiddenLayerT3[i].dSWCellIn = hiddenLayerT3[i].dSWCellIn*hiddenLayerT3[i].yForget + activationFunctionG(hiddenLayerT3[i].netCellState)*fPrime(hiddenLayerT3[i].netIn)*hiddenLayerT3[i].cellState;
		//partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
		hiddenLayerT3[i].dSWCellForget = hiddenLayerT3[i].dSWCellForget*hiddenLayerT3[i].yForget + hiddenLayerT3[i].previousCellState*fPrime(hiddenLayerT3[i].netForget)*hiddenLayerT3[i].previousCellState;
	}
}

// iL-wIH-hL-
//       /
//  rL-wRH
void recurrentPartialDerivatives(double* iL, LSTMWeight** wIH, LSTMCell* rL, LSTMWeight** wRH, LSTMCell* hL)
{
	//iL-wIH-hL
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			wIH[j][i].dSInputCell = wIH[j][i].dSInputCell*hL[i].yForget + gPrime(hL[i].netCellState)*hL[i].yIn*iL[j];
			wIH[j][i].dSInputInputGate = wIH[j][i].dSInputInputGate*hL[i].yForget + activationFunctionG(hL[i].netCellState)*fPrime(hL[i].netIn)*iL[j];
			wIH[j][i].dSInputForgetGate = wIH[j][i].dSInputForgetGate*hL[i].yForget + hL[i].previousCellState*fPrime(hL[i].netForget)*iL[j];
		}

		//internal connection
		hL[i].dSWCellIn = hL[i].dSWCellIn*hL[i].yForget + activationFunctionG(hL[i].netCellState)*fPrime(hL[i].netIn)*hL[i].cellState;
		hL[i].dSWCellForget = hL[i].dSWCellForget*hL[i].yForget + hL[i].previousCellState*fPrime(hL[i].netForget)*hL[i].previousCellState;
	}

	//rL-wRH-hL
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= HIDDEN; j++)
		{
			wRH[j][i].dSInputCell = wRH[j][i].dSInputCell*hL[i].yForget + gPrime(hL[i].netCellState)*hL[i].yIn*rL[j].cellOutput;
			wRH[j][i].dSInputInputGate = wRH[j][i].dSInputInputGate*hL[i].yForget + activationFunctionG(hL[i].netCellState)*fPrime(hL[i].netIn)*rL[j].cellOutput;
			wRH[j][i].dSInputForgetGate = wRH[j][i].dSInputForgetGate*hL[i].yForget + hL[i].previousCellState*fPrime(hL[i].netForget)*rL[j].cellOutput;
		}

		//internal connection
		hL[i].dSWCellIn += hL[i].dSWCellIn*hL[i].yForget + activationFunctionG(hL[i].netCellState)*fPrime(hL[i].netIn)*hL[i].cellState;
		hL[i].dSWCellForget += hL[i].dSWCellForget*hL[i].yForget + hL[i].previousCellState*fPrime(hL[i].netForget)*hL[i].previousCellState;
	}
}

void hObackwardPass(int index)
{	
	//for all output neurons
	for (int k = 0; k < OUTPUT; k++)
	{
		//output layer of linear neurons. find the difference between target and output
		//outputErrorGradients[k] = (inputData[index+1] - outputLayer[k]);
		outputErrorGradients[k] = (outputData[index] - outputLayer[k]);

		//for each connection to the hidden layer
		for (int j = 0; j <= HIDDEN; j++)
		{
			deltaHiddenOutput[j][k] += learningRate*hiddenLayerT3[j].cellOutput*outputErrorGradients[k];
		}
	}
}

void backwardPass(int index)
{	//backward pass


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
		hiddenLayerT3[j].gradientOutputGate = fPrime(hiddenLayerT3[j].netOut)*hiddenLayerT3[j].cellState*weightedSum;

		//internal cell state error
		hiddenLayerT3[j].cellStateError = hiddenLayerT3[j].yOut*weightedSum;
	}
	
	//output gates. for each connection to the hidden layer
	for (int i = 0; i < HIDDEN; i++)
	{
		//to the input layer
		for (int j = 0; j <= INPUT; j++)
		{
			//make the delta equal to the learning rate multiplied by the gradient multipled by the input for the connection
			iHWeightsT3[j][i].deltaOutputGateInput = learningRate*hiddenLayerT3[i].gradientOutputGate*inputLayerT3[j];
			iHWeightsT3[j][i].deltaInputGateInput = learningRate*hiddenLayerT3[i].cellStateError*iHWeightsT3[j][i].dSInputInputGate;
			iHWeightsT3[j][i].deltaForgetGateInput = learningRate*hiddenLayerT3[i].cellStateError*iHWeightsT3[j][i].dSInputForgetGate;
			iHWeightsT3[j][i].deltaInputCellInput = learningRate*hiddenLayerT3[i].cellStateError*iHWeightsT3[j][i].dSInputCell;
		}

		//for the internal connection
		hiddenLayerT3[i].deltaOutputGateCell = learningRate*hiddenLayerT3[i].gradientOutputGate*hiddenLayerT3[i].cellState;
		hiddenLayerT3[i].deltaInputGateCell = learningRate*hiddenLayerT3[i].cellStateError*hiddenLayerT3[i].dSWCellIn;
		hiddenLayerT3[i].deltaForgetGateCell = learningRate*hiddenLayerT3[i].cellStateError*hiddenLayerT3[i].dSWCellForget;
	}
}

void updateWeights(void)
{
	//updates weights for input to hidden layer
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j <= INPUT; j++)
		{
			//update connection weights
			iHWeightsT3[j][i].wInputCell += iHWeightsT3[j][i].deltaInputCellInput;
			iHWeightsT3[j][i].wInputInputGate += iHWeightsT3[j][i].deltaInputGateInput;
			iHWeightsT3[j][i].wInputForgetGate += iHWeightsT3[j][i].deltaForgetGateInput;
			iHWeightsT3[j][i].wInputOutputGate += iHWeightsT3[j][i].deltaOutputGateInput;
		}

		//update internal weights
		hiddenLayerT3[i].wCellIn += hiddenLayerT3[i].deltaInputGateCell;
		hiddenLayerT3[i].wCellForget += hiddenLayerT3[i].deltaForgetGateCell;
		hiddenLayerT3[i].wCellOut += hiddenLayerT3[i].deltaOutputGateCell;
	}

	//update weights for hidden to output layer
	for (int j = 0; j <= HIDDEN; j++)
	{
		for (int k = 0; k < OUTPUT; k++)
		{
			hOWeights[j][k] += deltaHiddenOutput[j][k];
		}
	}
}
