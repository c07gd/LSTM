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
	double cellStateError;

	//internal weights and deltas
	double wCellIn;
	double deltaInputGateCell;
	double wCellForget;
	double deltaForgetGateCell;
	double wCellOut;
	double deltaOutputGateCell;

	//partial derivatives
	double dSWCellIn;
	double dSWCellForget;
	//double dSWCellState;

	//output gate
	double netOut;
	double yOut;
	double gradientOutputGate;

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

	//partial derivatives. dont need partial derivative for output gate as it uses BP not RTRL
	double dSInputCell;
	double dSInputInputGate;
	double dSInputForgetGate;

	//deltas
	double deltaOutputGateInput;
	double deltaForgetGateInput;
	double deltaInputGateInput;
	double deltaInputCellInput;


	//functions
	LSTMWeight(void);
	void initialise(int iL);
};