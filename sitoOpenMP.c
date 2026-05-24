#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define vertices 16
#define CHUNK_SIZE 2000 // liczba pobieranych grafow
#define MAX_G6_LEN 64   // max dlugosc linii z grafem

typedef unsigned int matrix[vertices];

int eigensymmatrix_qr(int size, matrix original_graph, long double *values)
{
    long double A[vertices][vertices];
    long double Q[vertices][vertices];
    long double R[vertices][vertices];
    int i, j, k, iter;
    const int max_iter = 10000;
    const long double eps = 1e-12L;

    if (size <= 0)
        return 0;

    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < size; ++j)
        {
            if (i == j)
                A[i][j] = 0.0L;
            else
            {
                if (original_graph[i] & (1 << j))
                {
                    A[i][j] = 1.0L;
                }
                else
                {
                    A[i][j] = 0.0L;
                }
            }
        }
    }

    long double shift = size + 5.0L;
    for (i = 0; i < size; ++i)
        A[i][i] += shift;

    for (iter = 0; iter < max_iter; ++iter)
    {
        long double max_off = 0.0L;

        for (i = 0; i < size; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                Q[i][j] = A[i][j];
                R[i][j] = 0.0L;
            }
        }

        for (k = 0; k < size; ++k)
        {
            long double norm = 0.0L;
            for (i = 0; i < size; ++i)
                norm += Q[i][k] * Q[i][k];
            R[k][k] = sqrtl(norm);

            if (R[k][k] > 1e-18L)
            {
                for (i = 0; i < size; ++i)
                    Q[i][k] /= R[k][k];
            }

            for (j = k + 1; j < size; ++j)
            {
                long double dot = 0.0L;
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
                long double sum = 0.0L;
                for (int p = i; p < size; ++p)
                    sum += R[i][p] * Q[p][j];
                A[i][j] = sum;
                if (i != j && fabsl(sum) > max_off)
                    max_off = fabsl(sum);
            }
        }
        if (max_off < eps)
            break;
    }

    for (i = 0; i < size; ++i)
    {
        values[i + 1] = A[i][i] - shift;
    }
    return 1;
}

int isintegral(int N, long double *x)
{
    int i;
    long double u;
    for (i = 1; i <= N; i++)
    {
        u = x[i];
        if (!((ceill(u) - u < 1e-5L) || (u - floorl(u) < 1e-5L)))
            return 0;
    }
    return 1;
}

void BMKdecode(char *BUFFOR, int *N, matrix A)
{
    int bit = 32, poz = 1, i, j;
    *N = BUFFOR[0] - 63;

    for (i = 0; i < vertices; i++)
    {
        A[i] = 0;
    }

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

    long num_checked = 0;
    long num_found = 0;

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

// współdzielone zmienne to pliki wejściowe i wyjściowe oraz ilość sprawdzonych i znalezionych grafów
#pragma omp parallel default(none) shared(in, out, num_checked, num_found)
    {
        char local_chunk[CHUNK_SIZE][MAX_G6_LEN];
        int local_count;

        matrix A;
        long double values[17];
        int N;
        int i;

        while (1)
        {
            local_count = 0; // flaga dla danego wątku

// sekcja krytyczna (odczyt z pliku)
#pragma omp critical(odczyt_pliku)
            {
                for (i = 0; i < CHUNK_SIZE; i++)
                {
                    if (fgets(local_chunk[i], MAX_G6_LEN, in) != NULL)
                    {
                        local_count++;
                        num_checked++;
                    }
                    else
                    {
                        break; // koniec pliku
                    }
                }
            }

            // wszystkie dane zostały przeczytane koniec pracy
            if (local_count == 0)
            {
                break;
            }

            for (i = 0; i < local_count; i++)
            {
                BMKdecode(local_chunk[i], &N, A);
                eigensymmatrix_qr(N, A, values);

                if (isintegral(N, values))
                {
// sekcja krytyczna (zapis do pliku)
#pragma omp critical(zapis_pliku)
                    {
                        num_found++;
                        fprintf(out, "%s", local_chunk[i]);
                    }
                }
            }
        }
    }

    printf("Sprawdzono %ld grafow\n", num_checked);
    printf("Znaleziono %ld grafow\n", num_found);

    fclose(out);
    fclose(in);

    return 0;
}