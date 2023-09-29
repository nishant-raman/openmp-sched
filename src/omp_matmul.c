#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include "data.h"
#include <omp.h>
#include "timer.h"

#define chunk_size 1
#define kind 1

int main(int argc, char **argv)
{
    if(argc != 4){
        printf("Usage: <num_thread> <vec_a> <vec_b>.\n");
        exit(EXIT_FAILURE);
    }

    // int numThreads = omp_get_max_threads();
    // convinient for use in partitioning
    int numThreads = atoi(argv[1]);
    omp_set_num_threads(numThreads);
    omp_set_schedule(kind, chunk_size);
        
    struct timespec start;
    start_timer(&start);

    data_struct *d_1 = get_data_struct(argv[2]);
    data_struct *d_2 = get_data_struct(argv[3]);

    stop_timer(&start);
    fprintf(stderr, " (reading input)\n");

    if(d_1->cols != d_2->rows){
        printf("ERROR: Matrix dimension mismatch.\n");
        exit(EXIT_FAILURE);
    }

    data_struct *d_3 = malloc(sizeof(data_struct));
    d_3->rows = d_1->rows;
    d_3->cols = d_2->cols;

    d_3->data_point = calloc(d_3->rows, sizeof(double*));
    int i;
    for(i=0; i < d_3->rows; ++i)
        d_3->data_point[i]=calloc(d_3->cols, sizeof(double));

    start_timer(&start);
    /* Implementation of Matrix Multiplication  openMP */
int thread_cal = d_3->rows / numThreads;
int thread_cal_temp = 0;
if(thread_cal == 0){
    thread_cal_temp = d_3->cols / numThreads;
}

#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int x, y, z;
    if(thread_cal != 0){
    	for(x = tid*thread_cal; x < (tid+1)*thread_cal; x++) {
            for(y = 0; y < d_3->cols; y++) {
		d_3->data_point[x][y] = 0;
                for(z = 0; z < d_1->cols; z++){
                        d_3->data_point[x][y] += d_1->data_point[x][z] * d_2->data_point[z][y];
                }
            }
    	}
    }
    else if(thread_cal_temp != 0){
        for(x = 0; x < d_3->rows; x++) {
            for(y = tid*thread_cal_temp; y < (tid+1)*thread_cal_temp; y++) {
		d_3->data_point[x][y] = 0;
                for(z = 0; z < d_1->cols; z++){
                        d_3->data_point[x][y] += d_1->data_point[x][z] * d_2->data_point[z][y];
                }
            }
        }
    }

/* if matrix size is too small run it serial*/
    else{
	if(tid == 0){
        for(x = 0; x < d_3->rows; x++) {
            for(y = 0; y < d_3->cols; y++) {
		d_3->data_point[x][y] = 0;
                for(z = 0; z < d_1->cols; z++){
                        d_3->data_point[x][y] += d_1->data_point[x][z] * d_2->data_point[z][y];
                }
            }
        }
	}
    }
}

/* calculating the remainders 
    if(d_3->rows % numThreads != 0){
    	for(x = numThreads*thread_cal; x < d_3->rows; x++) {
            for(y = 0; y < d_3->cols; y++) {
                for(z = 0; z < d_1->cols; z++){
                        d_3->data_point[x][y] += d_1->data_point[x][z] * d_2->data_point[z][y];
                }
            }
    	}
    }
*/

    stop_timer(&start);
    fprintf(stderr, " (calculating answer)\n");
    
    start_timer(&start);
    /* Printing output */
	print_data(d_3);
    stop_timer(&start);
    fprintf(stderr, " (printing output)\n");
        free_data(d_1);
        free_data(d_2);
        free_data(d_3);
}
