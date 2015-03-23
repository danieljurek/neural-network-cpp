#include "NeuralNet.h"



neuralNet::neuralNet() {
	learnAlpha = .001; 		// Set learn alpha value to default.
}
 
void neuralNet::dimneuralnet(const size_t layer1, const size_t layer2, const size_t layer3) {
	inputLayer = new real [layer1]; 
	hiddenLayer = new real [layer2]; 
	outputLayer = new real [layer3];
	n1 = new real [layer1 * layer2];
	n2 = new real [layer2 * layer3]; 

	// Clear 2nd and 3rd node layers
	for (size_t i = 0; i < layer2; i++)
	{
		hiddenLayer[i] = 0; 
	}
	for (i = 0; i < layer3; i++)
	{
		outputLayer[i] = 0; 
	}
	
	numInput = layer1; 
	numHidden = layer2; 
	numOutput = layer3; 
}

 
void neuralNet::fillrandom(real DivideFactor) {
	sgenrand((unsigned long)time(0));

	for(size_t i = 0; i < numInput * numHidden; i++)
	{
		n1[i] = genrand() / DivideFactor; 
	}
	for(i = 0; i < numHidden * numOutput; i++)
	{
		n2[i] = genrand() / DivideFactor; 
	}

}

 
void neuralNet::fillfile(char *filename)
// Postcondition: neuralNet is filled with reals from the file. THIS FUNCTION ASSUMES THE FILE IS IN CORRECT FORMAT. 
{
	/* TEXT IMPLEMENTATION (READABLE AND SLOWER) 
	ifstream infile; 
	size_t layer1, layer2, layer3; 

	infile.open((const char *)filename); 		// Open the file
	infile >> layer1 >> layer2 >> layer3; 		// Read in the sizes
	dimneuralnet(layer1, layer2, layer3); 		// Dimension the neural net for those sizes

	// Fill the network with the values from the sizes
	for(size_t i = 0; i < numInput * numHidden; i++)
	{
		infile >> n1[i]; 
	}
	for (size_t i = 0; i < numHidden * numOutput; i++)
	{
		infile >> n2[i]; 
	}
	
	infile.close(); 
	*/
	// BINARY IMPLEMENTATION:
	float layer1, layer2, layer3; 		// Floats used to keep file size consistant, may be updated in a later version
	float temp; 
	FILE *inputfile = 0; 
	
	inputfile = fopen(filename, "rb"); 
	
	fread(&layer1, 1, sizeof(float), inputfile); 
	fread(&layer2, 1, sizeof(float), inputfile); 
	fread(&layer3, 1, sizeof(float), inputfile); 
	dimneuralnet((size_t)layer1, (size_t)layer2, (size_t)layer3); 
	
	for(size_t i = 0; i < numInput * numHidden; i++)
	{
		fread(&temp, 1, sizeof(float), inputfile); 
		n1[i] = temp; 
	}
	for(i = 0; i < numHidden * numOutput; i++)
	{
		fread(&temp, 1, sizeof(float), inputfile); 
		n2[i] = temp; 
	}
	
	fclose(inputfile); 
	
}

 
void neuralNet::savefile(char *filename)
// Postcondition: Outputs the neuralNet to a file in textural format
{
	/* TEXT IMPLEMENTATION (READABLE AND SLOWER) 
	ofstream outfile; 
	
	outfile.open((const char *)filename); 
	outfile << numInput << endl << numHidden << endl << numOutput << endl; // Output dimensions

	// Dump the rest of the network into the file. 
	for(size_t i = 0; i < numInput * numHidden; i++)
	{
		outfile << n1[i] << " "; 
	}	
	
	outfile << endl; 
	for(size_t i = 0; i < numHidden * numOutput; i++)
	{
		outfile << n2[i] << " "; 
	}
	
	outfile.close(); */
	
	// BINARY IMPLEMENTATION
	float temp; 
	FILE *outputfile = 0; 
	outputfile = fopen(filename, "wb"); 
	
	temp = numInput; 
	fwrite(&temp, sizeof(float), 1, outputfile); 
	temp = numHidden; 
	fwrite(&temp, sizeof(float), 1, outputfile); 
	temp = numOutput; 
	fwrite(&temp, sizeof(float), 1, outputfile); 
	
	for(size_t i = 0; i < numInput *numHidden; i++)
	{
		temp = n1[i]; 
		fwrite(&temp, sizeof(float), 1, outputfile); 
	}
	for(i = 0; i < numHidden * numOutput; i++)
	{
		temp = n2[i]; 
		fwrite(&temp, sizeof(float), 1, outputfile); 
	}
	
	fclose(outputfile); 
}

 
void neuralNet::input(real *inputdata) {
	for(size_t i = 0; i < numInput; i++)
	{
		inputLayer[i] = inputdata[i];  
	}
}

 
void neuralNet::process() {
	size_t i, j;
	
	// Propigate values from input to the hidden layer
	for(i = 0; i < numHidden; i++)
	{
		hiddenLayer[i] = 0;
		for(j = 0; j < numInput; j++)
		{
			hiddenLayer[i] += inputLayer[j] * n1[j + (i * numInput)]; 
		}
	}

	// Propigate values from hidden to output layer
	for(i = 0; i < numOutput; i++)
	{
		outputLayer[i] = 0; 
		for(j = 0; j < numHidden; j++)
		{	
			outputLayer[i] += hiddenLayer[j] * n2[j + (i * numHidden)]; 
		}
	}
	// Ok, that's it. We have output :) 
}

 
void neuralNet::result(real *&results) {
	results = new real [numOutput];

	for(size_t i = 0; i < numOutput; i++)
	{
		results[i] = outputLayer[i]; 
	}
}

 
void neuralNet::learn(real *desired) {

	real *outputError; 
	real *hiddenError; 
	real temp;  
	
	outputError = new real [numOutput]; 
	hiddenError = new real [numHidden]; 
	

	// Find error of the output nodes
	for(size_t i = 0; i < numOutput; i++)			// ERROR OUTPUT NODE
	{
		outputError[i] = (desired[i] - outputLayer[i]);

		// Other possible implementation 
		//outputError[i] *= outputLayer[i]* (1-outputLayer[i]); 
	}
	
	for(i = 0; i < numHidden; i++)					// ERROR HIDDEN LAYER
	{
		temp = 0; 
		for(size_t j = 0; j < numOutput; j++)
		{
			temp += outputError[j] * n2[i + (j * numHidden)]; 
		}
		
		hiddenError[i] = hiddenLayer[i] * temp;
	}


	// LEARN

	
	for(i = 0; i < numOutput; i++)					// HIDDEN -> OUTPUT
	{
		for(size_t j = 0; j < numHidden; j++)
		{
			n2[j + (i * numHidden)] += learnAlpha * hiddenLayer[j] * outputError[i]; 
		}
	}

	for(i = 0; i < numHidden; i++)				// INPUT -> HIDDEN
	{
		for(size_t j = 0; j < numInput; j++)
		{
			n1[j + (i * numInput)] += learnAlpha * inputLayer[j] * hiddenError[i]; 
		}
	}

}


void neuralNet::printNodes() {
	for(size_t i = 0; i < numOutput; i++)
	{
		cout << outputLayer[i] << " "; 
	}
	cout << endl; 
	for(i = 0; i < numHidden; i++)
	{
		cout << hiddenLayer[i] << " "; 
	}
	cout << endl; 
	for(i = 0; i < numInput; i++)
	{
		cout << inputLayer[i] << " "; 
	}
	cout << endl; 
}

 
void neuralNet::setLearnAlpha(real newLearnAlpha) {
	learnAlpha = newLearnAlpha; 
}


real neuralNet::getLearnAlpha() {
	return learnAlpha;
}


size_t neuralNet::getNumInputs() {
	return numInput; 
}


size_t neuralNet::getNumOutputs() {
	return numOutput; 
}
