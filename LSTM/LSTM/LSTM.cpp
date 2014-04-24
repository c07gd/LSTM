#include "LSTM.h"


//LSTM Cell class
LSTMCell::LSTMCell(void)
{
	//nothing here
}

void LSTMCell::initialise(bool type)
{
	if (type)
	{
		//input gate
		netIn = 0;
		yIn = 0;

		//forget gate
		netForget = 0;
		yForget = 0;

		//cell state
		netCellState = 0;
		previousCellState = 0; //this is important
		cellState = 0;
		cellStateError = 0;

		//internal weights, also important
		double internalRand = 1/sqrt(3);
		wCellIn = (((double)((rand()%100)+1)/100)*2*internalRand)-internalRand;
		deltaInputGateCell = 0;
		wCellForget = (((double)((rand()%100)+1)/100)*2*internalRand)-internalRand;
		deltaForgetGateCell = 0;
		wCellOut = (((double)((rand()%100)+1)/100)*2*internalRand)-internalRand;
		deltaOutputGateCell = 0;

		//partial derivatives
		dSWCellIn = 0;
		dSWCellForget = 0;
		//dSWCellState = 0;

		//output gate
		netOut = 0;
		yOut = 0;
		gradientOutputGate = 0;
		//deltaOutputGateInput = 0;

		//cell output
		cellOutput = 0;
	}
	else
	{
		//This seperate initialisation is so that the final cell is a bias cell.
		//input gate
		netIn = 0;
		yIn = 0;

		//forget gate
		netForget = 0;
		yForget = 0;

		//cell state
		netCellState = 0;
		previousCellState = 0; //this is important
		cellState = 0;
		cellStateError = 0;

		//internal weights
		wCellIn = 0;
		deltaInputGateCell = 0;
		wCellForget = 0;
		deltaForgetGateCell = 0;
		wCellOut = 0;
		deltaOutputGateCell = 0;

		//partial derivatives
		dSWCellIn = 0;
		dSWCellForget = 0;
		//dSWCellState = 0;

		//output gate
		netOut = 0;
		yOut = 0;
		gradientOutputGate = 0;

		//cell output
		cellOutput = -1;
	}
}





//LSTMWeight class
LSTMWeight::LSTMWeight(void)
{
	//nothing here
}

void LSTMWeight::initialise(int iL)
{
	//range of random values
	double inputHiddenRand = 1/sqrt(double(iL));

	//initialise each weight to random value
	wInputCell = (((double)((rand()%100)+1)/100)*2*inputHiddenRand)-inputHiddenRand;
	wInputInputGate = (((double)((rand()%100)+1)/100)*2*inputHiddenRand)-inputHiddenRand;
	wInputForgetGate = (((double)((rand()%100)+1)/100)*2*inputHiddenRand)-inputHiddenRand;
	wInputOutputGate = (((double)((rand()%100)+1)/100)*2*inputHiddenRand)-inputHiddenRand;

	//partial derivatives
	dSInputCell = 0;
	dSInputInputGate = 0;
	dSInputForgetGate = 0;	

	//deltas
	deltaInputCellInput = 0;
	deltaOutputGateInput = 0;
	deltaForgetGateInput = 0;
	deltaInputGateInput = 0;
}


