#include <stdio.h>
#include <stdlib.h>
#include "mnist_c.h"
#include <math.h>
#include <cuda.h>
#define input_size 6000
#define output_nodes 10
#define input_nodes 784

#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a,b) (((a)<(b))?(a):(b))

void shuffle(int *arr) {
    int n = input_size;
    for(int i = n-1; i >= 1; i--) {
        int j = rand() % (i+1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void w_initialisation(double *w_list, int w_list_size)
{
    for (int i = 0; i < w_list_size; i++)
        w_list[i] = ((double)rand() / (double)RAND_MAX) * 2 - 1;
        // w_list[i] = 0.31;
}

void initialise_with0 (double *list, int list_size)
{
    for (int i = 0; i < list_size; i++)
        list[i] = 0;
}

__device__ void initialise_with00 (double *list, int list_size)
{
    for (int i = 0; i < list_size; i++)
        list[i] = 0;
}

__device__ void sigmoid (int position, double *list, int list_size, double *list2)
{
    for (int i = position; i < position + list_size; i++)
        list[i] = 1.0 / (1.0 + exp(-list2[i]));
}
void sigmoid2 (int position, double *list, int list_size, double *list2)
{
    for (int i = position; i < position + list_size; i++)
        list[i] = 1.0 / (1.0 + exp(-list2[i]));
}

__device__ void sigmoid_dash (int position, double *list, int list_size, double *list2)
{
    for (int i = position; i < position + list_size; i++)
        list[i] = (1.0 / (1.0 + exp(-list2[i]))) * (1.0 - (1.0 / (1.0 + exp(-list2[i]))));
}

__device__ void matrix_mult(double *arr1, double *arr2, int dim1, int dim2, double *arr_res, int ctr1, int ctr2, int ctr1_prev)
{
    for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim2; k++)
            arr_res[j + ctr1] += arr1[j * dim2 + k + ctr2] * arr2[k + ctr1_prev];
}

void vector_add2(double *arr1, double *arr2, int dim, double *arr_res, int ctr)
{
    for (int i = 0 + ctr; i < ctr + dim; i++)
        arr_res[i] = arr1[i] + arr2[i];
}

void matrix_mult2(double *arr1, double *arr2, int dim1, int dim2, double *arr_res, int ctr1, int ctr2, int ctr1_prev)
{
    for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim2; k++)
            arr_res[j + ctr1] += arr1[j * dim2 + k + ctr2] * arr2[k + ctr1_prev];
}

__device__ void vector_add(double *arr1, double *arr2, int dim, double *arr_res, int ctr)
{
    for (int i = 0 + ctr; i < ctr + dim; i++)
        arr_res[i] = arr1[i] + arr2[i];
}

__device__ void dot_product(double *arr1, double *arr2, int dim, double *arr_res, int ctr)
{
    for (int i = 0; i < dim; i++)
        arr_res[i] = arr1[i] * arr2[i + ctr];
}

__device__ void matrix_mult_back(double *arr1, double *arr2, int dim1, int dim2, double *arr_res, int ctr1, int ctr2)
{
    for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim2; k++)
            arr_res[j * dim2 + k + ctr1] += arr1[j] * arr2[k + ctr2];
}

__device__ void matrix_mult_back2(double *arr1, double *arr2, int dim1, int dim2, double *arr_res, int ctr1, int ctr2)
{
    for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim2; k++)
            arr_res[j] += arr1[j + k * dim1 + ctr1] * arr2[k];
}

__device__ void Forward_prop(double *X, double y, int i, double *w_list, double *b_list, int nh, int nl, double *error, double *der_blist, double *der_wlist)
{
    int b_list_size = nh * nl + output_nodes;
    int w_list_size = (nh * input_nodes) + (nl - 1) * nh * nh + output_nodes * nh;
    double *der_blist_temp = (double*)malloc((b_list_size) * sizeof(double));
    double *der_wlist_temp = (double*)malloc((w_list_size) * sizeof(double));
    initialise_with00(der_blist_temp, b_list_size);
    initialise_with00(der_wlist_temp, w_list_size);
    double *a = (double*)malloc((nh * nl + output_nodes) * sizeof (double));
    double *z = (double*)malloc((nh * nl + output_nodes) * sizeof (double));
    double *del_error = (double*)malloc(output_nodes * sizeof(double));
    int j_ctr = 0, k_ctr = 0;
    initialise_with00 (z, nh * nl + output_nodes);
    initialise_with00 (a, nh * nl + output_nodes);

    matrix_mult(w_list, X, nh, input_nodes, z, j_ctr, k_ctr, 0);
    vector_add(z, b_list, nh, z, j_ctr);
    sigmoid(j_ctr, a, nh, z);
    int prev_j_ctr = j_ctr;
    j_ctr += nh;
    k_ctr += input_nodes * nh;

    for (int j = 0; j < nl - 1; j++)
    {
        matrix_mult(w_list, a, nh, nh, z, j_ctr, k_ctr, prev_j_ctr);
        vector_add(z, b_list, nh, z, j_ctr);
        sigmoid(j_ctr, a, nh, z);
        prev_j_ctr = j_ctr;
        j_ctr += nh;
        k_ctr += nh * nh;
    }


    matrix_mult(w_list, a, output_nodes, nh, z, j_ctr, k_ctr, prev_j_ctr);
    vector_add(z, b_list, output_nodes, z, j_ctr);
    sigmoid(j_ctr, a, output_nodes, z);

    double this_error = 0;


    for (int j = 0; j < output_nodes; j++)
    {
        if ((int)y == j)
            del_error[j] = a[j + j_ctr] - 1;
        else
            del_error[j] = a[j + j_ctr];
        this_error += pow(del_error[j], 2);
    //    printf("%f ", del_error[j]);
    }
    // printf("\n");
    atomicAdd((double*) &error[0], this_error);

    // Back prop
    j_ctr = nh * nl;
    k_ctr =  (nh * input_nodes) + (nl - 1) * nh * nh;
    sigmoid_dash(j_ctr, z, output_nodes, z);
    double *del_l2 = (double*)malloc((output_nodes) * sizeof (double));
    dot_product(del_error, z, output_nodes, del_l2, j_ctr);
    for (int j = 0; j < output_nodes; j++)
    {
        der_blist_temp[j_ctr + j] += del_l2[j];
    }
    
    j_ctr -= nh;
    matrix_mult_back(del_l2, a, output_nodes, nh, der_wlist_temp, k_ctr, j_ctr);

    double *del_l1 = (double*)malloc((nh) * sizeof (double));

    int ctr = 1;
    double *temp;

    

    // for (int k = 0; k < nl - 1; k++)
    // {
    //     initialise_with00(del_l1, nh);
    //     sigmoid_dash(j_ctr, z, nh, z);
    //     if (ctr == 1)
    //         matrix_mult_back2(w_list, del_l2, nh, output_nodes, del_l1, k_ctr, j_ctr);
    //     else
    //         matrix_mult_back2(w_list, del_l2, nh, nh, del_l1, k_ctr, j_ctr);
    //     dot_product(del_l1, z, nh, del_l1, j_ctr);
    
    //     k_ctr -= nh * nh;

    //     for (int j = 0; j < nh; j++)
    //         der_blist_temp[j + j_ctr] += del_l1[j];

    //     j_ctr -= nh;
    //     matrix_mult_back(del_l1, a, nh, nh, der_wlist_temp, k_ctr, j_ctr);

    //     if (ctr == 1)
    //     {
    //         del_l2 = (double*)realloc(del_l2, sizeof(double) * nh);
    //         ctr = 0;
    //     }
    //     temp = del_l1;
    //     del_l1 = del_l2;
    //     del_l2 = temp;
    // }

    // double *del_l1 = (double*)malloc((nh) * sizeof (double));
    initialise_with00(del_l1, nh);
    sigmoid_dash(j_ctr, z, nh, z);
    matrix_mult_back2(w_list, del_l2, nh, output_nodes, del_l1, k_ctr, j_ctr);
    dot_product(del_l1, z, nh, del_l1, j_ctr);
    k_ctr = 0;
    for (int j = 0; j < nh; j++)
    {
        der_blist_temp[j + j_ctr] += del_l1[j];
    }

    matrix_mult_back(del_l1, X, nh, input_nodes, der_wlist_temp, k_ctr, j_ctr);

    for (int j = 0; j < w_list_size; j++)
    {
        atomicAdd((double*) &der_wlist[j], der_wlist_temp[j]);
        // printf("%f ", der_wlist[j]);
    }

    for (int j = 0; j < b_list_size; j++)
    {
        atomicAdd((double*) &der_blist[j], der_blist_temp[j]);
        // printf("%f ", der_blist[j]);
    }


    free(del_l2);
    free(del_l1);
    free(a);
    free(z);
    free(del_error);
    free(der_blist_temp);
    free(der_wlist_temp);
}

__global__ void Main_loop(double *train_image, double *train_label, int loop_cnt, int nb, int *selection_list, double *w_list, double *b_list, int nh, int nl, double *error, double *der_blist, double *der_wlist)
{
    int w_list_size = (nh * input_nodes) + (nl - 1) * nh * nh + output_nodes * nh;
    // printf("%i\n", SIZE);
    int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = loop_cnt * nb + tid0; i < (loop_cnt + 1) * nb; i += blockDim.x * gridDim.x)
    {
        Forward_prop(&train_image[selection_list[i] * SIZE], train_label[selection_list[i]], selection_list[i], w_list, b_list, nh, nl, error, der_blist, der_wlist);
    }
}

int main(int argc, char** argv)
{
    // srand ( time(NULL) );
    load_mnist();

    cudaEvent_t                /* CUDA timers */
    start_device,
    stop_device;  
    float time_device;
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    double alpha = 0.1;

    int nl = atoi(argv[1]);
    int nh = atoi(argv[2]);
    int ne = atoi(argv[3]);
    int nb = atoi(argv[4]);

    int b_list_size = nh * nl + output_nodes;
    int w_list_size = (nh * input_nodes) + (nl - 1) * nh * nh + output_nodes * nh;
    double *b_list = (double*)malloc((b_list_size) * sizeof(double));
    double *w_list = (double*)malloc((w_list_size) * sizeof(double));
    double *der_blist = (double*)malloc((b_list_size) * sizeof(double));
    double *der_wlist = (double*)malloc((w_list_size) * sizeof(double));

    int *selection_list = (int*)malloc(input_size * sizeof(int));
    for (int i = 0; i < input_size; i++)
        selection_list[i] = i;

    double *train_image_h = (double*)malloc((SIZE * NUM_TRAIN) * sizeof(double));
    double *test_image_h = (double*)malloc((SIZE * NUM_TRAIN) * sizeof(double));

    for (int i = 0; i < NUM_TRAIN; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            train_image_h[i * SIZE + j] = train_image[i][j];
        }
    }

    for (int i = 0; i < NUM_TEST; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            test_image_h[i * SIZE + j] = test_image[i][j];
        }
    }

    double *train_image_c, *train_label_c, *w_list_c, *b_list_c, *der_wlist_c, *der_blist_c;
    int *selection_list_c;
    cudaMalloc((void **) &train_image_c, (SIZE * NUM_TRAIN)*sizeof(double));
    cudaMalloc((void **) &train_label_c, (SIZE * NUM_TRAIN)*sizeof(double));
    cudaMalloc((void **) &selection_list_c, (input_size)*sizeof(int));
    cudaMalloc((void **) &w_list_c, (w_list_size)*sizeof(double));
    cudaMalloc((void **) &b_list_c, (b_list_size)*sizeof(double));
    cudaMalloc((void **) &der_wlist_c, (w_list_size)*sizeof(double));
    cudaMalloc((void **) &der_blist_c, (b_list_size)*sizeof(double));

    cudaMemcpy(train_image_c,train_image_h,(SIZE * NUM_TRAIN)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(train_label_c, train_label, (NUM_TRAIN) *sizeof(double), cudaMemcpyHostToDevice);

    w_initialisation(w_list, w_list_size);
    initialise_with0(b_list, b_list_size);  

    int nblocks, nthreads_per_block, nt;
    nthreads_per_block = nb;
    nblocks = 1;
    nt = nb;

    double *error_c;

    double *error = (double*)malloc((1) * sizeof(double));
    cudaMalloc((void **) &error_c, (1)*sizeof(double));

    double prev_cum_error = 0, cum_error = 0;
    int loop_cnt = 0;
    int batches = input_size / nb;

    cudaEventRecord( start_device, 0 ); 
    for (int j = 0; j < ne; j++)
    {
        shuffle(selection_list);
        cudaMemcpy(selection_list_c, selection_list, (input_size) *sizeof(int), cudaMemcpyHostToDevice);
        loop_cnt = 0;
        cum_error = 0;
        for (int l = 0; l < batches; l++)
        {
            error[0] = 0;
            cudaMemcpy(error_c, error, (1) *sizeof(double), cudaMemcpyHostToDevice);
            initialise_with0(der_blist, b_list_size);
            initialise_with0(der_wlist, w_list_size);
            cudaMemcpy(w_list_c, w_list,(w_list_size)*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(b_list_c, b_list, (b_list_size) *sizeof(double), cudaMemcpyHostToDevice);  
            cudaMemcpy(der_wlist_c, der_wlist, (w_list_size) *sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(der_blist_c, der_blist, (b_list_size) *sizeof(double), cudaMemcpyHostToDevice);  
            Main_loop<<<nblocks, nthreads_per_block>>>(train_image_c, train_label_c, loop_cnt, nb, selection_list_c, w_list_c, b_list_c, nh, nl, error_c, der_blist_c, der_wlist_c);
            cudaDeviceSynchronize();
            cudaMemcpy(error, error_c, (1) *sizeof(double), cudaMemcpyDeviceToHost);
            error[0] = error[0] / nb / 2.0;
            loop_cnt += 1;
            cum_error += error[0];

            // for (int k = 0; k < 1; k++)
            // {
            //     printf("%f ", der_wlist_c[k]);
            // }

            cudaMemcpy(der_blist, der_blist_c, b_list_size * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(der_wlist, der_wlist_c, w_list_size * sizeof(double), cudaMemcpyDeviceToHost);
            for (int k = 0; k < w_list_size; k++)
            {
                w_list[k] -= (alpha * der_wlist[k] / nb);
            }
            for (int k = 0; k < b_list_size; k++)
                b_list[k] -= (alpha * der_blist[k] / nb);
        }
        cum_error /= loop_cnt;

        if (j % 10 == 0)
            printf("Error: %f\n", cum_error);
    }

    cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    cudaEventElapsedTime( &time_device, start_device, stop_device );

    printf("time elapsed device: %f(s)\n",  time_device/1000.);

    int cnt = 0;
    for (int i = 0; i < input_size; i++)
    {
        double *a = (double*)malloc((nh * nl + output_nodes) * sizeof (double));
        double *z = (double*)malloc((nh * nl + output_nodes) * sizeof (double));
        int j_ctr = 0, k_ctr = 0;
        initialise_with0 (z, nh * nl + output_nodes);
        initialise_with0 (a, nh * nl + output_nodes);

        matrix_mult2(w_list, &train_image_h[i * SIZE], nh, input_nodes, z, j_ctr, k_ctr, 0);
        vector_add2(z, b_list, nh, z, j_ctr);
        sigmoid2(j_ctr, a, nh, z);
        int prev_j_ctr = j_ctr;
        j_ctr += nh;
        k_ctr += input_nodes * nh;

        for (int j = 0; j < nl - 1; j++)
        {
            matrix_mult2(w_list, a, nh, nh, z, j_ctr, k_ctr, prev_j_ctr);
            vector_add2(z, b_list, nh, z, j_ctr);
            sigmoid2(j_ctr, a, nh, z);
            prev_j_ctr = j_ctr;
            j_ctr += nh;
            k_ctr += nh * nh;
        }

        matrix_mult2(w_list, a, output_nodes, nh, z, j_ctr, k_ctr, prev_j_ctr);
        vector_add2(z, b_list, output_nodes, z, j_ctr);
        sigmoid2(j_ctr, a, output_nodes, z);

        double max = 0;
        int max_index = -1;
        for (int j = 0; j < output_nodes; j++)
        {   
            if (a[j + j_ctr] > max)
            {
                max = a[j + j_ctr];
                max_index = j;
            }
        }
        if (max_index == (int)train_label[i])
            cnt += 1;

        free(a);
        free(z);
    }

    double accuracy = (double)cnt / (double)input_size;
    printf("Accuracy: %f\n", accuracy);


    cnt = 0;
    for (int i = 0; i < 10000; i++)
    {
        double *a = (double*)malloc((nh * nl + output_nodes) * sizeof (double));
        double *z = (double*)malloc((nh * nl + output_nodes) * sizeof (double));
        int j_ctr = 0, k_ctr = 0;
        initialise_with0 (z, nh * nl + output_nodes);
        initialise_with0 (a, nh * nl + output_nodes);

        matrix_mult2(w_list, test_image[i], nh, input_nodes, z, j_ctr, k_ctr, 0);
        vector_add2(z, b_list, nh, z, j_ctr);
        sigmoid2(j_ctr, a, nh, z);
        int prev_j_ctr = j_ctr;
        j_ctr += nh;
        k_ctr += input_nodes * nh;

        for (int j = 0; j < nl - 1; j++)
        {
            matrix_mult2(w_list, a, nh, nh, z, j_ctr, k_ctr, prev_j_ctr);
            vector_add2(z, b_list, nh, z, j_ctr);
            sigmoid2(j_ctr, a, nh, z);
            prev_j_ctr = j_ctr;
            j_ctr += nh;
            k_ctr += nh * nh;
        }

        matrix_mult2(w_list, a, output_nodes, nh, z, j_ctr, k_ctr, prev_j_ctr);
        vector_add2(z, b_list, output_nodes, z, j_ctr);
        sigmoid2(j_ctr, a, output_nodes, z);

        double max = 0;
        int max_index = -1;
        for (int j = 0; j < output_nodes; j++)
        {   
            if (a[j + j_ctr] > max)
            {
                max = a[j + j_ctr];
                max_index = j;
            }
        }
        if (max_index == (int)test_label[i])
            cnt += 1;

        free(a);
        free(z);
    }

    accuracy = (double)cnt / (double)10000;
    printf("Test Accuracy: %f\n", accuracy);


    for (int i = 0; i < NUM_TRAIN; i++)
    {
        free(train_image[i]);
        free(train_image_char[i]);
        free(train_label_char[i]);
    }

    for (int i = 0; i < NUM_TEST; i++)
    {
        free(test_image[i]);
        free(test_image_char[i]);
        free(test_label_char[i]);
    }

    // free(a);
    // free(z);
    free(error);
    cudaFree(error_c);
    cudaFree(w_list_c);
    cudaFree(b_list_c);
    cudaFree(der_wlist_c);
    cudaFree(der_blist_c);
    cudaFree(selection_list_c);
    cudaFree(train_image_c);
    cudaFree(train_label_c);
    free(train_image_h);
    free(test_image_h);
    free(der_blist);
    free(der_wlist);
    free(selection_list);
    free(w_list);
    free(b_list);
    free(train_image);
    free(test_image);
    free(train_label);
    free(test_label);
    return 0;
}

