
#include <stdio.h>
#include <time.h>			// necessary for random numbers
#include <math.h>
#include <iostream> 
#include "random.h"

using namespace std; 

typedef float real;			// Only use random numbers

class neuralNet
{
	
	public:
		neuralNet();								
		
		void dimneuralnet(const size_t layer1, const size_t layer2, const size_t layer3);		// dimension network & nodes
		void fillrandom(real DivideFactor);					// fill network w/ random weights numbers
		void fillfile(char *filename);							// fill network from file
		void savefile(char *filename); 							// save the network to a file
	
		void input(real *inputdata);								// give a value to the input nodes
		void process(); 														// Perform the neural network calculation
		void learn(real *desired); 									// teach the neural network based on an array of reals of desired output
	
		void result(real *&results); 								// fill array with results 
		void printNodes();													// give console output of the nodes
		
		void setLearnAlpha(real newLearnAlpha); 		// set the learnAlpha value 
		real getLearnAlpha(); 											// returns the learnAlpha value
		size_t getNumInputs(); 											// returns the number of input nodes
		size_t getNumOutputs(); 										// returns the number of output nodes 
	
	private:
		real *inputLayer;				// layer 1
		real *hiddenLayer;			// layer 2	 
		real *outputLayer;			// layer 3
		
		real *n1;								// weights between later 1 and 2
		real *n2;								// weights between layer 2 and 3
		
		size_t numInput, numHidden, numOutput;    		// keep track of the number of elements
		
		double learnAlpha; 			// alpha multiplier for learning
};
