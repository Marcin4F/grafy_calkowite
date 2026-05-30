#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define vertices 16
#define CHUNK_SIZE 100000
#define MAX_G6_LEN 64

typedef unsigned int matrix[vertices];

//  na karcie graficznej
__global__ void eigensymmatrix_qr(matrix *d_graphs, int *d_N_array, int *d_results, int num_graphs)
{
    // id wątku
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_graphs)
        return;

    int size = d_N_array[tid];
    if (size <= 0 || size > vertices)
    {
        d_results[tid] = 0;
        return;
    }

    double A[vertices][vertices];
    double Q[vertices][vertices];
    double R[vertices][vertices];
    int i, j, k, iter;
    const int max_iter = 1000;
    const double eps = 1e-10;

    // kopiowanie z pamięci globalnej do prywatnych rejestrów
    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < size; ++j)
        {
            if (i == j)
                A[i][j] = 0.0;
            else
                A[i][j] = (d_graphs[tid][i] & (1 << j)) ? 1.0 : 0.0;
        }
    }

    double shift = size + 5.0;
    for (i = 0; i < size; ++i)
        A[i][i] += shift;

    // QR
    for (iter = 0; iter < max_iter; ++iter)
    {
        double max_off = 0.0;
        for (i = 0; i < size; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                Q[i][j] = A[i][j];
                R[i][j] = 0.0;
            }
        }
        for (k = 0; k < size; ++k)
        {
            double norm = 0.0;
            for (i = 0; i < size; ++i)
                norm += Q[i][k] * Q[i][k];
            R[k][k] = sqrt(norm);
            if (R[k][k] > 1e-18)
            {
                for (i = 0; i < size; ++i)
                    Q[i][k] /= R[k][k];
            }
            for (j = k + 1; j < size; ++j)
            {
                double dot = 0.0;
                for (i = 0; i < size; ++i)
                    dot += Q[i][k] * Q[i][j];
                R[k][j] = dot;
                for (i = 0; i < size; ++i)
                    Q[i][j] -= dot * Q[i][k];
            }
        }
        for (i = 0; i < size; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                double sum = 0.0;
                for (int p = i; p < size; ++p)
                    sum += R[i][p] * Q[p][j];
                A[i][j] = sum;
                if (i != j && fabs(sum) > max_off)
                    max_off = fabs(sum);
            }
        }
        if (max_off < eps)
            break;
    }

    // isintegral przeniesione na gpu
    int is_int = 1;
    for (i = 0; i < size; ++i)
    {
        double val = A[i][i] - shift;
        if (!((ceil(val) - val < 1e-5) || (val - floor(val) < 1e-5)))
        {
            is_int = 0;
            break;
        }
    }

    // 1 - całkowy, 0 - niecałkowy
    d_results[tid] = is_int;
}

void BMKdecode(char *BUFFOR, int *N, matrix A)
{
    int bit = 32, poz = 1, i, j;
    *N = BUFFOR[0] - 63;
    for (i = 0; i < vertices; i++)
        A[i] = 0;
    for (i = 1; i < *N; i++)
    {
        for (j = 0; j < i; j++)
        {
            if (bit == 0)
            {
                bit = 32;
                poz++;
            }
            if ((BUFFOR[poz] - 63) & bit)
            {
                A[i] |= (1 << j);
                A[j] |= (1 << i);
            }
            bit = bit >> 1;
        }
    }
}

int main(int argc, char *argv[])
{
    FILE *in, *out;

    if (argc < 3)
    {
        printf("Za mało parametrów\n");
        return -1;
    }

    in = fopen(argv[1], "r");
    if (in == NULL)
    {
        printf("Brak pliku zrodlowego\n");
        return -1;
    }

    out = fopen(argv[2], "wb");
    if (out == NULL)
    {
        printf("Błąd pliku do zapisu");
        fclose(in);
        return -1;
    }

    // CPU
    char (*h_strings)[MAX_G6_LEN] = (char (*)[MAX_G6_LEN])malloc(CHUNK_SIZE * MAX_G6_LEN);
    matrix *h_graphs = (matrix *)malloc(CHUNK_SIZE * sizeof(matrix));
    int *h_N_array = (int *)malloc(CHUNK_SIZE * sizeof(int));
    int *h_results = (int *)malloc(CHUNK_SIZE * sizeof(int));

    // GPU
    matrix *d_graphs;
    int *d_N_array;
    int *d_results;
    cudaMalloc(&d_graphs, CHUNK_SIZE * sizeof(matrix));
    cudaMalloc(&d_N_array, CHUNK_SIZE * sizeof(int));
    cudaMalloc(&d_results, CHUNK_SIZE * sizeof(int));

    long num_checked = 0;
    long num_found = 0;
    int local_count = 0;

    printf("Rozpoczeto przetwarzanie na GPU (CUDA)...\n");

    while (1)
    {
        local_count = 0;

        // odczyt i dekodowanie grafów
        while (local_count < CHUNK_SIZE && fgets(h_strings[local_count], MAX_G6_LEN, in))
        {
            BMKdecode(h_strings[local_count], &h_N_array[local_count], h_graphs[local_count]);
            local_count++;
            num_checked++;
        }

        if (local_count == 0)
            break;

        // kopiowanie do karty
        cudaMemcpy(d_graphs, h_graphs, local_count * sizeof(matrix), cudaMemcpyHostToDevice);
        cudaMemcpy(d_N_array, h_N_array, local_count * sizeof(int), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (local_count + threadsPerBlock - 1) / threadsPerBlock;
        eigensymmatrix_qr<<<blocksPerGrid, threadsPerBlock>>>(d_graphs, d_N_array, d_results, local_count);
        cudaDeviceSynchronize();

        // pobranie wyników
        cudaMemcpy(h_results, d_results, local_count * sizeof(int), cudaMemcpyDeviceToHost);

        // zapis do pliku
        for (int i = 0; i < local_count; i++)
        {
            if (h_results[i] == 1)
            {
                num_found++;
                fprintf(out, "%s", h_strings[i]);
            }
        }
    }

    printf("Sprawdzono %ld grafow\n", num_checked);
    printf("Znaleziono %ld grafow\n", num_found);

    free(h_strings);
    free(h_graphs);
    free(h_N_array);
    free(h_results);
    cudaFree(d_graphs);
    cudaFree(d_N_array);
    cudaFree(d_results);

    fclose(out);
    fclose(in);

    return 0;
}