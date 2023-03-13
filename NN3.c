#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include <math.h>
#include <time.h>
#include "timer.h"
#define input_size 20000
#define output_nodes 10
#define input_nodes 784

struct timeval timerStart;

void StartTimer(){
  gettimeofday(&timerStart, NULL);
}

double GetTimer(){
  struct timeval timerStop, timerElapsed;
  gettimeofday(&timerStop, NULL);
  timersub(&timerStop, &timerStart, &timerElapsed);

  return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}


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

void sigmoid (int position, double *list, int list_size, double *list2)
{
    for (int i = position; i < position + list_size; i++)
        list[i] = 1.0 / (1.0 + exp(-list2[i]));
}

void sigmoid_dash (int position, double *list, int list_size, double *list2)
{
    for (int i = position; i < position + list_size; i++)
        list[i] = (1.0 / (1.0 + exp(-list2[i]))) * (1.0 - (1.0 / (1.0 + exp(-list2[i]))));
}

void matrix_mult(double *arr1, double *arr2, int dim1, int dim2, double *arr_res, int ctr1, int ctr2, int ctr1_prev)
{
    for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim2; k++)
            arr_res[j + ctr1] += arr1[j * dim2 + k + ctr2] * arr2[k + ctr1_prev];
}

void vector_add(double *arr1, double *arr2, int dim, double *arr_res, int ctr)
{
    for (int i = 0 + ctr; i < ctr + dim; i++)
        arr_res[i] = arr1[i] + arr2[i];
}

void dot_product(double *arr1, double *arr2, int dim, double *arr_res, int ctr)
{
    for (int i = 0; i < dim; i++)
        arr_res[i] = arr1[i] * arr2[i + ctr];
}

void matrix_mult_back(double *arr1, double *arr2, int dim1, int dim2, double *arr_res, int ctr1, int ctr2)
{
    for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim2; k++)
            arr_res[j * dim2 + k + ctr1] += arr1[j] * arr2[k + ctr2];
}

void matrix_mult_back2(double *arr1, double *arr2, int dim1, int dim2, double *arr_res, int ctr1, int ctr2)
{
    for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim2; k++)
            arr_res[j] += arr1[j + k * dim1 + ctr1] * arr2[k];
}

void Forward_prop(double *X, double y, int i, double *w_list, double *b_list, int nh, int nl, double *error, double *a, double *z, double *del_error)
{
    int j_ctr = 0, k_ctr = 0;
    initialise_with0 (z, nh * nl + output_nodes);
    initialise_with0 (a, nh * nl + output_nodes);

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
        // printf("%f ", del_error[j]);
    }
    // printf("\n");
    *error += this_error;
}


void Backward_prop(double *der_blist, double *der_wlist, double *del_error, double *a, double *z, double *w_list, double *X, int nl, int nh)
{
    int j_ctr = nh * nl;
    int k_ctr =  (nh * input_nodes) + (nl - 1) * nh * nh;
    sigmoid_dash(j_ctr, z, output_nodes, z);
    double *del_l2 = (double*)malloc((output_nodes) * sizeof (double));
    dot_product(del_error, z, output_nodes, del_l2, j_ctr);
    for (int j = 0; j < output_nodes; j++)
    {
        der_blist[j_ctr + j] += del_l2[j];
        // printf("%f ", del_l2[j]);
    }
    // printf("\n");
    
    j_ctr -= nh;
    matrix_mult_back(del_l2, a, output_nodes, nh, der_wlist, k_ctr, j_ctr);

    double *del_l1 = (double*)malloc((nh) * sizeof (double));

    int ctr = 1;
    double *temp;

    for (int k = 0; k < nl - 1; k++)
    {
        initialise_with0(del_l1, nh);
        sigmoid_dash(j_ctr, z, nh, z);
        if (ctr == 1)
            matrix_mult_back2(w_list, del_l2, nh, output_nodes, del_l1, k_ctr, j_ctr);
        else
            matrix_mult_back2(w_list, del_l2, nh, nh, del_l1, k_ctr, j_ctr);
        dot_product(del_l1, z, nh, del_l1, j_ctr);
    
        k_ctr -= nh * nh;

        for (int j = 0; j < nh; j++)
            der_blist[j + j_ctr] += del_l1[j];

        j_ctr -= nh;
        matrix_mult_back(del_l1, a, nh, nh, der_wlist, k_ctr, j_ctr);

        if (ctr == 1)
        {
            del_l2 = (double*)realloc(del_l2, sizeof(double) * nh);
            ctr = 0;
        }
        temp = del_l1;
        del_l1 = del_l2;
        del_l2 = temp;
    }

    // double *del_l1 = (double*)malloc((nh) * sizeof (double));
    initialise_with0(del_l1, nh);
    sigmoid_dash(j_ctr, z, nh, z);
    matrix_mult_back2(w_list, del_l2, nh, output_nodes, del_l1, k_ctr, j_ctr);
    dot_product(del_l1, z, nh, del_l1, j_ctr);
    k_ctr = 0;
    for (int j = 0; j < nh; j++)
    {
        der_blist[j + j_ctr] += del_l1[j];
    }

    matrix_mult_back(del_l1, X, nh, input_nodes, der_wlist, k_ctr, j_ctr);

    // int b_list_size = nh * nl + output_nodes;
    // for (int j = 0; j < b_list_size; j++)
    // {
    //     printf("%f ", der_blist[j]);
    // }

    free(del_l2);
    free(del_l1);
}

int main(int argc, char** argv)
{
    srand ( time(NULL) );
    load_mnist();

    double alpha = 0.1;

    int nl = atoi(argv[1]);
    int nh = atoi(argv[2]);
    int ne = atoi(argv[3]);
    int nb = atoi(argv[4]);

    int b_list_size = nh * nl + output_nodes;
    int w_list_size = (nh * input_nodes) + (nl - 1) * nh * nh + output_nodes * nh;
    double *b_list = (double*)malloc((b_list_size) * sizeof(double));
    double *w_list = (double*)malloc((w_list_size) * sizeof(double));
    double *del_error = (double*)malloc(output_nodes * sizeof(double));
    double *der_blist = (double*)malloc((b_list_size) * sizeof(double));
    double *der_wlist = (double*)malloc((w_list_size) * sizeof(double));
    double *a = (double*)malloc((nh * nl + output_nodes) * sizeof (double));
    double *z = (double*)malloc((nh * nl + output_nodes) * sizeof (double));

    w_initialisation(w_list, w_list_size);
    initialise_with0(b_list, b_list_size);

    int *selection_list = (int*)malloc(input_size * sizeof(int));
    for (int i = 0; i < input_size; i++)
        selection_list[i] = i;

    double prev_cum_error = 0, error = 0, cum_error = 0;
    int loop_cnt = 0;
    int batches = input_size / nb;

    StartTimer();

    for (int j = 0; j < ne; j++)
    {
        shuffle(selection_list);
        loop_cnt = 0;
        cum_error = 0;
        for (int l = 0; l < batches; l++)
        {
            error = 0;
            initialise_with0(der_blist, b_list_size);
            initialise_with0(der_wlist, w_list_size);
            for (int i = loop_cnt * nb; i < (loop_cnt + 1) * nb; i++)
            {
                Forward_prop(train_image[selection_list[i]], train_label[selection_list[i]], selection_list[i], w_list, b_list, nh, nl, &error, a, z, del_error);
                Backward_prop(der_blist, der_wlist, del_error, a, z, w_list, train_image[selection_list[i]], nl, nh);
            }

            error = error / nb / 2.0;
            loop_cnt += 1;
            cum_error += error;

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
        
        if (cum_error < 0.03)
            break;
    }
    
    const double tElapsed = GetTimer() / 1000.0;
    printf("TotalTime: %1.1fs\n", tElapsed);

    int cnt = 0;
    for (int i = 0; i < input_size; i++)
    {
        int j_ctr = 0, k_ctr = 0;
        initialise_with0 (z, nh * nl + output_nodes);
        initialise_with0 (a, nh * nl + output_nodes);

        matrix_mult(w_list, train_image[i], nh, input_nodes, z, j_ctr, k_ctr, 0);
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
    }

    double accuracy = (double)cnt / (double)input_size;
    printf("Accuracy: %f\n", accuracy);



    cnt = 0;
    for (int i = 0; i < 10000; i++)
    {
        int j_ctr = 0, k_ctr = 0;
        initialise_with0 (z, nh * nl + output_nodes);
        initialise_with0 (a, nh * nl + output_nodes);

        matrix_mult(w_list, test_image[i], nh, input_nodes, z, j_ctr, k_ctr, 0);
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
    }

    accuracy = (double)cnt / 10000.0;
    printf("Test Accuracy: %f\n", accuracy);

    for (int i = 0; i < NUM_TRAIN; i++)
        free(train_image[i]);

    for (int i = 0; i < NUM_TEST; i++)
        free(test_image[i]);

    free(a);
    free(z);
    free(der_blist);
    free(der_wlist);
    free(del_error);
    free(selection_list);
    free(w_list);
    free(b_list);
    free(train_image);
    free(test_image);
    free(train_label);
    free(test_label);
    return 0;
}