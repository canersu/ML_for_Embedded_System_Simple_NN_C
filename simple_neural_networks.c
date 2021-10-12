#include "simple_neural_networks.h"
#include <math.h>

double single_in_single_out_nn(double  input, double weight) {
	return input*weight;
}


double weighted_sum(double * input, double * weight, uint32_t INPUT_LEN) {
	double output = 0;
	for(int i=0; i<INPUT_LEN; i++)
	{
		output+=single_in_single_out_nn(input[i], weight[i]);
	}
 return output;
}


double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN) {
	double predicted_value = 0;
	predicted_value = weighted_sum(input, weight, INPUT_LEN);
	return predicted_value;
}


void elementwise_multiple( double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN) {
	for(int i=0; i<VECTOR_LEN; ++i)
	{
		output_vector[i] = input_scalar * weight_vector[i];
	}
}


void single_input_multiple_output_nn(double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN){
  elementwise_multiple(input_scalar, weight_vector,output_vector,VECTOR_LEN);
}


void matrix_vector_multiplication(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	double row_tot;

	for(int i=0; i< OUTPUT_LEN; ++i)
	{
		row_tot = 0.0;
		for(int j=0; j<INPUT_LEN; ++j)
		{
			row_tot += weights_matrix[i][j]*input_vector[j];
		}
		output_vector[i] = row_tot;
	}
}


void multiple_inputs_multiple_outputs_nn(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	matrix_vector_multiplication(input_vector,INPUT_LEN,output_vector,OUTPUT_LEN,weights_matrix);
}


void hidden_nn( double *input_vector, uint32_t INPUT_LEN,
				uint32_t HIDDEN_LEN, double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
				uint32_t OUTPUT_LEN, double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN], double *output_vector) {

	double hidden[3] = {0,0,0};
	matrix_vector_multiplication(input_vector, INPUT_LEN, hidden, HIDDEN_LEN, in_to_hid_weights);
	matrix_vector_multiplication(hidden, HIDDEN_LEN, output_vector, OUTPUT_LEN, hid_to_out_weights);

}


double find_error(double yhat, double y) {
	double error = powf((yhat-y),2);
	return error;
}


void brute_force_learning( double input, double weight, double expected_value, double step_amount, uint32_t itr) {
   double prediction,error;
   double up_prediction, down_prediction, up_error, down_error;
   int i;
	 for(i=0;i<itr;i++){

		 prediction  = input * weight;
		 error = find_error(prediction,expected_value);

		 printf("Step: %d   Error: %f    Prediction: %f    Weight: %f\n", i, error, prediction, weight);

		 up_prediction =  input * (weight + step_amount);
		 up_error      =   powf((up_prediction - expected_value),2);
		 down_prediction =  input * (weight - step_amount);
		 down_error      =  powf((down_prediction - expected_value),2);

		 if(down_error <  up_error)
			   weight = weight + up_error;
		 if(down_error >  up_error)
			   weight = weight + down_error;
	 }
}


void linear_forward_nn(double *input_vector, uint32_t INPUT_LEN,
						double *output_vector, uint32_t OUTPUT_LEN,
						double weights_matrix[OUTPUT_LEN][INPUT_LEN], double *weights_b) {

	matrix_vector_multiplication(input_vector,INPUT_LEN, output_vector,OUTPUT_LEN,weights_matrix);

	for(int k=0;k<OUTPUT_LEN;k++){
		output_vector[k]+=weights_b[k];
	}
}


double relu(double x){
	if(x>0){return x;}
	else{return 0;}
}


void vector_relu(double *input_vector, double *output_vector, uint32_t LEN) {
	  for(int i =0;i<LEN;i++){
		  output_vector[i] =  relu(input_vector[i]);
		}
}


double sigmoid(double x) {
	 double result =  0;
	 result = (1)/(1+exp(-1*x));
	 return result;
}


void vector_sigmoid(double * input_vector, double * output_vector, uint32_t LEN) {
	for (int i = 0; i < LEN; i++) {
		output_vector[i] = sigmoid(input_vector[i]);
	}
}


double compute_cost(uint32_t m, double yhat[m][1], double y[m][1]) {
	double cost = 0;
	for(int i=0; i<m; ++i)
	{
		cost += ((y[i][0]*log(yhat[i][0])) + ((1-y[i][0])*(log(1-yhat[i][0]))));
	}
	cost = -1*cost/m;

	return cost;
}


void normalize_data_2d(uint32_t ROW, uint32_t COL, double input_matrix[ROW][COL], double output_matrix[ROW][COL]){
	double max =  -99999999;
	for(int i =0;i<ROW;i++){
	  for(int j =0;j<COL;j++){
		  if(input_matrix[i][j] >max){
			  max = input_matrix[i][j];
			}
		}
	}

	for(int i=0;i<ROW;i++){
		for(int j=0;j<COL;j++){
	    output_matrix[i][j] =  input_matrix[i][j]/max;
		}
	}
}


// Use this function to print matrix values for debugging
void matrix_print(uint32_t ROW, uint32_t COL, double A[ROW][COL]) {
	for(int i=0; i<ROW; i++){
			for(int j=0; j<COL; j++){
				printf(" %f ", A[i][j]);
			}
			printf("\n");
	}
	printf("\n\r");
}


void weights_random_initialization(uint32_t HIDDEN_LEN, uint32_t INPUT_LEN, double weight_matrix[HIDDEN_LEN][INPUT_LEN]) {
	double d_rand;

	/*Seed random number generator*/
	srand(1);

	for (int i = 0; i < HIDDEN_LEN; i++) {
		for (int j = 0; j < INPUT_LEN; j++) {
			/*Generate random numbers between 0 and 1*/
			d_rand = (rand() % 10);
			d_rand /= 10;
			weight_matrix[i][j] = d_rand;
		}
	}
}


void zero_initialization(uint32_t HIDDEN_LEN, uint32_t INPUT_LEN, double weight_matrix[HIDDEN_LEN][INPUT_LEN]) {
	memset(weight_matrix, 0, sizeof(weight_matrix));
	for (int i = 0; i < HIDDEN_LEN; i++) {
		for (int j = 0; j < INPUT_LEN; j++) {
			weight_matrix[i][j] = 0;
		}
	}
}


void weightsB_zero_initialization(double * weightsB, uint32_t LEN){
	memset(weightsB, 0, LEN*sizeof(weightsB[0]));
}


void relu_backward(uint32_t m, uint32_t LAYER_LEN, double dA[m][LAYER_LEN], double Z[m][LAYER_LEN], double dZ[m][LAYER_LEN]) {
	for(int i=0; i<m; ++i)
	{
		for(int j=0; j<LAYER_LEN; ++j)
		{
			if(dA[i][j] >= 0)
			{
				elementwise_multiple(1, dA[i], dZ[i], LAYER_LEN);
			}
			else{
				elementwise_multiple(0, dA[i], dZ[i], LAYER_LEN);
			}
		}
	}
}


void linear_backward(uint32_t LAYER_LEN, uint32_t PREV_LAYER_LEN, uint32_t m, double dZ[m][LAYER_LEN],
		double A_prev[m][PREV_LAYER_LEN], double dW[LAYER_LEN][PREV_LAYER_LEN], double * db ){
	double tot = 0.0;
	double a_prev_t[PREV_LAYER_LEN][1];
	zero_initialization(PREV_LAYER_LEN, 1, a_prev_t);

	for(int i=0; i<m; ++i)
	{

		double dw_iter[LAYER_LEN][PREV_LAYER_LEN];
		zero_initialization(LAYER_LEN, PREV_LAYER_LEN, dw_iter);
		matrix_transpose(1, PREV_LAYER_LEN, A_prev[i], a_prev_t);
		matrix_matrix_multiplication(LAYER_LEN, 1, PREV_LAYER_LEN, dZ[i], a_prev_t, dw_iter);
		matrix_multiply_scalar(LAYER_LEN, PREV_LAYER_LEN, (double) 1/m, dw_iter, dw_iter);
		matrix_matrix_sum(LAYER_LEN, PREV_LAYER_LEN, dw_iter, dW, dW);

	}
	// Find db
	for(int i=0; i<m; ++i)
	{
		for(int j=0; j<LAYER_LEN; ++j)
		{
			tot += dZ[i][j];
			db[j] = (double) 1/m*tot;
		}
	}
}


void matrix_matrix_sum(uint32_t MATRIX_ROW, uint32_t MATRIX_COL,
									double input_matrix1[MATRIX_ROW][MATRIX_COL],
									double input_matrix2[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; ++c) {
	      for (int d = 0; d < MATRIX_COL; ++d) {
	        output_matrix[c][d] = input_matrix1[c][d]+input_matrix2[c][d];
	      }
	 }
}


void matrix_divide_scalar(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double scalar,
									double input_matrix[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; c++) {
	      for (int d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix[c][d]/scalar;
	      }
	 }
}


void matrix_matrix_multiplication(uint32_t MATRIX1_ROW, uint32_t MATRIX1_COL, uint32_t MATRIX2_COL,
									double input_matrix1[MATRIX1_ROW][MATRIX1_COL],
									double input_matrix2[MATRIX1_COL][MATRIX2_COL],
									double output_matrix[MATRIX1_ROW][MATRIX2_COL]) {

	for(int k=0;k<MATRIX1_ROW;k++){
		 memset(output_matrix[k], 0, MATRIX2_COL*sizeof(output_matrix[0][0]));
	}
	double sum=0;
	for (int c = 0; c < MATRIX1_ROW; c++) {
	      for (int d = 0; d < MATRIX2_COL; d++) {
	        for (int k = 0; k < MATRIX1_COL; k++) {
	          sum += input_matrix1[c][k]*input_matrix2[k][d];
	        }
	        output_matrix[c][d] = sum;
	        sum = 0;
	      }
	 }
}


void matrix_matrix_sub(uint32_t MATRIX_ROW, uint32_t MATRIX_COL,
									double input_matrix1[MATRIX_ROW][MATRIX_COL],
									double input_matrix2[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; c++) {
	      for (int d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix1[c][d]-input_matrix2[c][d];
	      }
	 }
}


void weights_update(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double learning_rate,
									double dW[MATRIX_ROW][MATRIX_COL],
									double W[MATRIX_ROW][MATRIX_COL]) {
	double out[MATRIX_ROW][MATRIX_COL];
	matrix_multiply_scalar(MATRIX_ROW, MATRIX_COL, learning_rate, dW, out);
	matrix_matrix_sub(MATRIX_ROW, MATRIX_COL, W, out, W);
}


void matrix_multiply_scalar(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double scalar,
									double input_matrix[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; c++) {
	      for (int d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix[c][d]*scalar;
	      }
	 }
}


void matrix_transpose(uint32_t ROW, uint32_t COL, double A[ROW][COL], double A_T[COL][ROW]) {
	for(int i=0; i<ROW; i++){
		for(int j=0; j<COL; j++){
			A_T[j][i]=A[i][j];
		}
	}
}
