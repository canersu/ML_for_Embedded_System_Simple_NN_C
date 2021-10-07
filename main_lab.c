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
#define NUM_OF_HID1_NODES	3
#define NUM_OF_OUT_NODES	1	// output classes

double learning_rate=0.01;

/*Input layer to hidden layer*/
double a1[1][NUM_OF_HID1_NODES];	// activation function
double b1[NUM_OF_HID1_NODES];		// bias
double z1[1][NUM_OF_HID1_NODES];	// output vector

// Input layer to hidden layer weight matrix
double w1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] =    {{0.25, 0.5,   0.05},   	 //hid[0]
													{0.8,  0.82,  0.3 },     //hid[1]
													{0.5,  0.45,  0.19}};   //hid[2]

/*Hidden layer to output layer*/
double b2[NUM_OF_OUT_NODES];
double z2[1][NUM_OF_OUT_NODES];	// Predicted output vector

// Hidden layer to output layer weight matrix
double w2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES] =    {{0.48, 0.73, 0.03}};

// Predicted values
double yhat[1][NUM_OF_OUT_NODES];
double yhat_eg[NUM_OF_OUT_NODES];	// Predicted yhat

// Training data
double train_x[1][NUM_OF_FEATURES];				// Training data after normalization
double train_y[1][NUM_OF_OUT_NODES] = {{1}};  	// The expected (training) y values


void main(void) {
	// Raw training data

	double raw_x[1][NUM_OF_FEATURES] = {{23.0, 40.0, 100.0}};	// temp, hum, air_q input values,


	normalize_data_2d(NUM_OF_FEATURES,1, raw_x, train_x);	// Data normalization
	printf("train_x \n");
	matrix_print(1, NUM_OF_FEATURES, train_x);

	weightsB_zero_initialization(b1, NUM_OF_HID1_NODES);
	weightsB_zero_initialization(b2, NUM_OF_OUT_NODES);

	// Lab 3.1
	linear_forward_nn(train_x, NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, b1);
	printf("Output vector (Z1_1): %f\n", z1[0][0]);
	printf("Output vector (Z1_2): %f\n", z1[0][1]);
	printf("Output vector (z1_3): %f\r\n", z1[0][2]);

	vector_relu(z1[0],a1,NUM_OF_HID1_NODES);
	printf("relu_a \n");
	matrix_print(1, NUM_OF_HID1_NODES, a1);

	linear_forward_nn(a1, NUM_OF_HID1_NODES, z2[0], NUM_OF_OUT_NODES, w2, b2);
	printf("Output vector (Z2): %f\r\n", z2[0][0]);

	/*compute yhat*/
	vector_sigmoid(z2[0],yhat[0], NUM_OF_OUT_NODES);
	printf("yhat:  %f\n\r", yhat[0][0]);

	double cost = compute_cost(1, yhat, train_y);
	printf("cost:  %f\r\n", cost);

	// Lab 4.1 - backpropagation
	double dA1[1][NUM_OF_HID1_NODES] = {{0, 0, 0}};
	double dA2[1][1] = {{0}};

	double dZ1[1][NUM_OF_HID1_NODES] = {{0, 0, 0}};
	double dZ2[1][1] = {{0}};

	double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	double dW2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES] = {{0, 0, 0}};

	double db1[NUM_OF_HID1_NODES] = {0, 0, 0};
	double db2[NUM_OF_OUT_NODES] = {0};

	/* Output layer */

	//dZ2 = A2-Y = yhat - y
	//TODO: calculate dZ2, use matrix_matrix_sub() function
	matrix_matrix_sub(1,3,yhat,train_y, dZ2);

	printf("dZ2 \n");
	matrix_print(1, 1, dZ2);
	double a1_t[3][1] = {{0}, {0}, {0}};
	matrix_transpose(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, a1, a1_t);


	printf("a1_t  \n");
	matrix_print( NUM_OF_HID1_NODES, 1, a1_t);

	// TODO: Calculate linear backward for output layer, use linear_backward() function
	//check for formula on slide 31 (lecture 5)
	linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, 1, dZ2, a1_t, dW2, db2);

	printf("dW2 \n");
	matrix_print(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, dW2);

	printf("db2 \n");
	matrix_print(NUM_OF_OUT_NODES, 1, db2);


	double W2_T[NUM_OF_HID1_NODES][NUM_OF_OUT_NODES] = {{0},{0},{0}};

	// TODO: Make matrix transpose for output layer, use matrix_transpose() function
	matrix_transpose(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, w2, W2_T);

	printf("W2_T \n");
	matrix_print(NUM_OF_HID1_NODES, NUM_OF_OUT_NODES, W2_T);

	// TODO: Make matrix matrix multiplication; use matrix_matrix_multiplication() function
	// Check for formula on slide 31 (lecture 5)
	matrix_matrix_multiplication(NUM_OF_HID1_NODES, NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, W2_T, dZ2, dA1);
	//relu_backward(1, NUM_OF_HID1_NODES, dA1, z1, dZ1);


	printf("dA1 \n");
	matrix_print(1, NUM_OF_HID1_NODES, dA1);

	/* Input layer */

	// TODO: Calculate relu backward for hidden layer, use relu_backward() function
	// Check for formula on slide 31 (lecture 5)
	relu_backward(1, NUM_OF_HID1_NODES, dA1, z1, dZ1);


	printf("dZ1 \n");
	matrix_print(1, NUM_OF_HID1_NODES, dZ1);

	// TODO: Calculate linear backward for hidden layer, use linear_backward() function
	// Check for formula on slide 31 (lecture 5)
	double a0_t[3][1];
	matrix_transpose(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, train_x, a0_t);
	linear_backward(NUM_OF_HID1_NODES, NUM_OF_FEATURES, 1, dZ1, a0_t, dW1, db1);
	printf("a0_t  \n");
	matrix_print( 3, 1, a0_t);

	printf("w1_1 \n");
	matrix_print(3, NUM_OF_HID1_NODES, w1);

	printf("dW1  \n");
	matrix_print( NUM_OF_HID1_NODES, NUM_OF_FEATURES, dW1);
	/*UPDATE PARAMETERS*/

	// W1 = W1 - learning_rate * dW1
	// TODO: update weights for W1, use weights_update() function
	weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, dW1, w1);

	printf("updated W1  \n");
	matrix_print( NUM_OF_HID1_NODES, NUM_OF_FEATURES, w1);

	// b1 = b1 - learning_rate * db1
	// TODO: update bias for b1, use weights_update() function
	weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, db1, b1);


	printf("updated b1  \n");
	matrix_print(NUM_OF_HID1_NODES, 1, b1);

	// W2 = W2 - learning_rate * dW2
	// TODO: update weights for W2, use weights_update() function
	weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, dW2, w2);

	printf("updated W2  \n");
	matrix_print( NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, w2);


	// b2 = b2 - learning_rate * db2
	// TODO: update bias for b2, use weights_update() function
	weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, db2, b2);


	printf("updated b2  \n");
	matrix_print( NUM_OF_OUT_NODES, 1, b2);


	/*PREDICT*/
	printf("-------- PREDICT --------\n");
	double input_x_eg[1][NUM_OF_FEATURES] = {{20, 40, 110}};
	double input_x[1][NUM_OF_FEATURES] = {{0, 0, 0}};

	normalize_data_2d(1,1, input_x_eg, input_x);

	/*compute z1*/
	linear_forward_nn(input_x, NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, b1);

	/*compute a1*/
	vector_relu(z1[0],a1[0],NUM_OF_HID1_NODES);

	/*compute z2*/
	linear_forward_nn(a1[0], NUM_OF_HID1_NODES, z2[0], NUM_OF_OUT_NODES, w2, b2);
	printf("z2_eg1:  %f \n",z2[0][0]);

	/*compute yhat*/
	vector_sigmoid(z2[0],yhat_eg, NUM_OF_OUT_NODES);
	printf("predicted:  %f\n\r", yhat_eg[0]);

}

