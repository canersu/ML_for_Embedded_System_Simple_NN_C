/*
 ============================================================================
 Name        : Lab4_backpropagation.c
 Author      : Mairo Leier
 Version     :
 Copyright   : TalTech
 Description : Backpropagation in C, Ansi-style

 If you get error during compilation:
 	 "undefined reference to `powf'	simple_neural_networks.c	/Lab3_2_PC_find_error/src	line 72	C/C++ Problem"
 Add: Project Properties -> C/C++ Build -> Settings -> Tool Settings -> GCC C Linker -> Miscellanous -> Linker flags: add "-lm -E"
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"

// Size of the layers
#define NUM_OF_FEATURES   	3  	// input values
#define NUM_OF_HID1_NODES	5
#define NUM_OF_HID2_NODES	4
#define NUM_OF_OUT_NODES	1	// output classes
#define NUM_SAMPLES         10

double learning_rate=0.01;

/*Input layer to hidden layer*/
double a1[NUM_SAMPLES][NUM_OF_HID1_NODES];	// activation function
double b1[NUM_OF_HID1_NODES];		        // bias
double z1[NUM_SAMPLES][NUM_OF_HID1_NODES];	// output vector

// Input layer to hidden layer1 weight matrix
double w1[NUM_OF_HID1_NODES][NUM_OF_FEATURES];;


/*Hidden layer to output layer*/
double a2[NUM_SAMPLES][NUM_OF_HID2_NODES];	// activation function
double b2[NUM_OF_HID2_NODES];
double z2[NUM_SAMPLES][NUM_OF_HID2_NODES];	// Predicted output vector

// Hidden layer to output layer weight matrix
double w2[NUM_OF_HID1_NODES][NUM_OF_HID2_NODES];


/*Hidden layer to output layer*/
double b3[NUM_OF_OUT_NODES];
double z3[NUM_SAMPLES][NUM_OF_OUT_NODES];	// Predicted output vector

// Hidden layer to output layer weight matrix
double w3[NUM_OF_HID2_NODES][NUM_OF_OUT_NODES];

// Predicted values
double yhat[NUM_SAMPLES][NUM_OF_OUT_NODES];
double yhat_eg[NUM_OF_OUT_NODES];	// Predicted yhat

// Training data
double train_x[NUM_SAMPLES][NUM_OF_FEATURES];				                                    // Training data after normalization
double train_y[NUM_SAMPLES][NUM_OF_OUT_NODES] = {{1},{1},{1},{0},{1},{0},{0},{0},{0},{1}};  	// The expected (training) y values


void main(void) {
	// Raw training data

	double raw_x[NUM_SAMPLES][NUM_OF_FEATURES] = {{23.0, 40.0, 100.0},	// temp, hum, air_q input values,
										 {15.0, 60.0, 50.0},
										 {2.0,  35.0, 30.0},
										 {30.0, 52.0, 180.0},
										 {22.0, 95.0, 70.0},
										 {35.0, 97.0, 400.0},
										 {60.0, 20.0, 200.0},
										 {85.0, 50.0, 90.0},
										 {27.0, 10.0,  300.0},
										 {17.0, 70.0, 20.0}};

	normalize_data_2d(NUM_OF_FEATURES,NUM_SAMPLES, raw_x, train_x);	// Data normalization
	printf("train_x \n");
	matrix_print(NUM_SAMPLES, NUM_OF_FEATURES, train_x);

	// Random weight initialization
	weights_random_initialization(NUM_OF_HID1_NODES, NUM_OF_FEATURES, w1);
	weights_random_initialization(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, w2);
	weights_random_initialization(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, w3);

	// Zero bias initialization
	weightsB_zero_initialization(b1, NUM_OF_HID1_NODES);
	weightsB_zero_initialization(b2, NUM_OF_HID2_NODES);
	weightsB_zero_initialization(b3, NUM_OF_OUT_NODES);


	for(int epoch=0; epoch<10; ++epoch)
	{
		// Lab 3.1
		for(int i=0; i<NUM_SAMPLES; ++i)
		{
			linear_forward_nn(train_x[i], NUM_OF_FEATURES, z1[i], NUM_OF_HID1_NODES, w1, b1);

			vector_relu(z1[i],a1[i],NUM_OF_HID1_NODES);
		}
		printf("relu_a1 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_HID1_NODES, a1);

		for(int i=0; i<NUM_SAMPLES; ++i)
		{
			linear_forward_nn(a1[i], NUM_OF_HID1_NODES, z2[i], NUM_OF_HID2_NODES, w2, b2);
			vector_relu(z2[i],a2[i],NUM_OF_HID2_NODES);
		}
		printf("relu_a2 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_HID2_NODES, a2);

		/*compute yhat*/
		for(int i=0; i<NUM_SAMPLES; ++i)
		{
			linear_forward_nn(a2[i], NUM_OF_HID2_NODES, z3[i], NUM_OF_OUT_NODES, w3, b3);
			vector_sigmoid(z3[i],yhat[i], NUM_OF_OUT_NODES);
		}
		double cost = compute_cost(NUM_SAMPLES, yhat, train_y);
		printf("cost:  %f\r\n", cost);

		// Backpropagation

		// Initialize reverse layer output variables
		double dA1[NUM_SAMPLES][NUM_OF_HID1_NODES];
		zero_initialization(NUM_SAMPLES, NUM_OF_HID1_NODES, dA1);
		double dA2[NUM_SAMPLES][NUM_OF_HID2_NODES];
		zero_initialization(NUM_SAMPLES, NUM_OF_HID2_NODES, dA2);
		double dA3[NUM_SAMPLES][NUM_OF_OUT_NODES];
		zero_initialization(NUM_SAMPLES, NUM_OF_OUT_NODES, dA3);

		double dZ1[NUM_SAMPLES][NUM_OF_HID1_NODES];
		zero_initialization(NUM_SAMPLES, NUM_OF_HID1_NODES, dZ1);
		double dZ2[NUM_SAMPLES][NUM_OF_HID2_NODES];
		zero_initialization(NUM_SAMPLES, NUM_OF_HID2_NODES, dZ2);
		double dZ3[NUM_SAMPLES][NUM_OF_OUT_NODES];
		zero_initialization(NUM_SAMPLES, NUM_OF_OUT_NODES, dZ3);

		double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES];
		zero_initialization(NUM_OF_HID1_NODES, NUM_OF_FEATURES, dW1);
		double dW2[NUM_OF_HID2_NODES][NUM_OF_HID1_NODES];
		zero_initialization(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, dW2);
		double dW3[NUM_OF_OUT_NODES][NUM_OF_HID2_NODES];
		zero_initialization(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, dW3);

		double db1[NUM_OF_HID1_NODES];
		weightsB_zero_initialization(db1, NUM_OF_HID1_NODES);
		double db2[NUM_OF_HID2_NODES];
		weightsB_zero_initialization(db2, NUM_OF_HID2_NODES);
		double db3[NUM_OF_OUT_NODES];
		weightsB_zero_initialization(db3, NUM_OF_OUT_NODES);

		/* Output layer */

		// Calculate dZ3
		matrix_matrix_sub(NUM_SAMPLES, NUM_OF_OUT_NODES, yhat, train_y, dZ3);

		printf("dZ3 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_OUT_NODES, dZ3);

		// Calculate linear backward for output layer
		linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, NUM_SAMPLES, dZ3, a2, dW3, db3);

		printf("dW3 \n");
		matrix_print(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, dW3);

		printf("db3 \n");
		matrix_print(NUM_OF_OUT_NODES, 1, db3);

		printf("dZ3 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_OUT_NODES, dZ3);


		// Define and initialize W3 transpose
		double W3_T[NUM_OF_HID2_NODES][NUM_OF_OUT_NODES];
		zero_initialization(NUM_OF_HID2_NODES, NUM_OF_OUT_NODES, W3_T);
		matrix_transpose(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, w3, W3_T);
		printf("W3_T \n");
		matrix_print(NUM_OF_HID2_NODES, NUM_OF_OUT_NODES, W3_T);

		// Make matrix matrix multiplication
		for(int i=0; i<NUM_SAMPLES; ++i)
		{
			matrix_vector_multiplication(dZ3[i], NUM_OF_OUT_NODES, dA2[i], NUM_OF_HID2_NODES, W3_T);
		}

		printf("dA2 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_HID2_NODES, dA2);

		/* Input layer */

		// Calculate relu backward for hidden layer, use relu_backward() function
		relu_backward(NUM_SAMPLES, NUM_OF_HID2_NODES, dA2, z2, dZ2);


		printf("dZ2 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_HID2_NODES, dZ2);

        // Calculate linear backward for hidden layer, use linear_backward() function
		linear_backward(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, NUM_SAMPLES, dZ2, a1, dW2, db2);


		printf("dW2  \n");
		matrix_print(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, dW2);

		printf("dZ2 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_HID2_NODES, dZ2);

		// Define and initialize W2 transpose
		double W2_T[NUM_OF_HID1_NODES][NUM_OF_HID2_NODES];
		zero_initialization(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, W2_T);
		matrix_transpose(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, w2, W2_T);
		printf("W2_T \n");
		matrix_print(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, W2_T);

		for(int i=0; i<NUM_SAMPLES; ++i)
		{
			matrix_vector_multiplication(dZ2[i], NUM_OF_HID2_NODES, dA1[i], NUM_OF_HID1_NODES, W2_T);
		}


		relu_backward(NUM_SAMPLES, NUM_OF_HID1_NODES, dA1, z1, dZ1);


		printf("dA1 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_HID1_NODES, dA1);

		printf("dZ1 \n");
		matrix_print(NUM_SAMPLES, NUM_OF_HID1_NODES, dZ1);

		printf("train_x \n");
		matrix_print(NUM_SAMPLES, NUM_OF_FEATURES, train_x);



		linear_backward(NUM_OF_HID1_NODES, NUM_OF_FEATURES, NUM_SAMPLES, dZ1, train_x, dW1, db1);
		printf("dW1  \n");
		matrix_print(NUM_OF_HID1_NODES, NUM_OF_FEATURES, dW1);

		// Define and initialize W1 transpose
		double W1_T[NUM_OF_FEATURES][NUM_OF_HID1_NODES];
		zero_initialization(NUM_OF_FEATURES, NUM_OF_HID1_NODES, W1_T);
		matrix_transpose(NUM_OF_HID1_NODES, NUM_OF_FEATURES, w1, W1_T);
		printf("W1_T \n");
		matrix_print(NUM_OF_FEATURES, NUM_OF_HID1_NODES, W1_T);

		for(int i=0; i<NUM_SAMPLES; ++i)
		{
			matrix_vector_multiplication(dZ1[i], NUM_OF_HID1_NODES, train_x[i], NUM_OF_FEATURES, W1_T);
		}

		/*UPDATE PARAMETERS*/

		// W1 = W1 - learning_rate * dW1
		weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, dW1, w1);

		printf("updated W1  \n");
		matrix_print( NUM_OF_HID1_NODES, NUM_OF_FEATURES, w1);

		// b1 = b1 - learning_rate * db1
		weights_update(NUM_OF_HID1_NODES, 1, learning_rate, db1, b1);
		printf("updated b1  \n");
		matrix_print(NUM_OF_HID1_NODES, 1, b1);

		// W2 = W2 - learning_rate * dW2
		weights_update(NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, learning_rate, dW2, w2);
		printf("updated W2  \n");
		matrix_print( NUM_OF_HID2_NODES, NUM_OF_HID1_NODES, w2);

		// b2 = b2 - learning_rate * db2
		weights_update(NUM_OF_HID1_NODES, 1, learning_rate, db2, b2);
		printf("updated b2  \n");
		matrix_print(NUM_OF_HID2_NODES, 1, b2);

		// W3 = W3 - learning_rate * dW3
		weights_update(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, learning_rate, dW3, w3);
		printf("updated W3  \n");
		matrix_print( NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, w3);

		// b2 = b2 - learning_rate * db2
		weights_update(NUM_OF_OUT_NODES, NUM_OF_HID2_NODES, learning_rate, db3, b3);
		printf("updated b3  \n");
		matrix_print(NUM_OF_OUT_NODES, 1, b3);

		printf("------------------END OF EPOCH %d------------------\n",epoch);
	}
	/*PREDICT*/
	printf("-------- PREDICT --------\n");
	double input_x_eg[1][NUM_OF_FEATURES] = {{20, 40, 80}};
	double input_x[1][NUM_OF_FEATURES] = {{0, 0, 0}};

	normalize_data_2d(NUM_OF_FEATURES,1, input_x_eg, input_x);
	printf("input_x \n");
	matrix_print(1, NUM_OF_FEATURES, input_x);

	/*compute z1*/
	linear_forward_nn(input_x[0], NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, b1);

	/*compute a1*/
	vector_relu(z1[0],a1[0],NUM_OF_HID1_NODES);

	/*compute z2*/
	linear_forward_nn(a1[0], NUM_OF_HID1_NODES, z2[0], NUM_OF_HID2_NODES, w2, b2);

	/*compute a2*/
	vector_relu(z2[0],a2[0],NUM_OF_HID2_NODES);

	/*compute z3*/
	linear_forward_nn(a2[0], NUM_OF_HID2_NODES, z3[0], NUM_OF_OUT_NODES, w3, b3);
	printf("z3_eg1:  %f \n",z3[0][0]);

	/*compute yhat*/
	vector_sigmoid(z3[0],yhat_eg, NUM_OF_OUT_NODES);
	printf("predicted:  %f\n\r", yhat_eg[0]);

}

