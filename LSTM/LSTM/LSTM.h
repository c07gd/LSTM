#include <iostream>
//LSTM header

class LSTMCell
{
public:

	//input gate
	double netIn;
	double yIn;

	//forget gate
	double netForget;
	double yForget;

	//cell state
	double netCellState;
	double previousCellState;
	double cellState;

	//internal weights
	double wCellIn;
	double wCellForget;
	double wCellOut;

	//output gate
	double netOut;
	double yOut;

	//cell output
	double cellOutput;

	LSTMCell(void);
	void initialise(bool type);
};

class LSTMWeight
{
public:

	//variables
	double wInputCell;
	double wInputInputGate;
	double wInputForgetGate;
	double wInputOutputGate;

	//functions
	LSTMWeight(void);
	void initialise(int iL);
};