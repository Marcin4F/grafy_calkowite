#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define vertices 16
typedef unsigned int matrix[vertices];

// algorytm iteracji QR z przesunięcien do obliczania widma grafu
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

    // kopiowanie z macierzy bitowej do matematycznej z wykorzystaniem masek bitowych
    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < size; ++j)
        {
            if (i == j)
            {
                A[i][j] = 0.0L; // przekątna
            }
            else
            {
                // najstarszy bit to najmniejszy indeks
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

    // spectrum shift (N + 5), na koniec odejmowane
    long double shift = size + 5.0L;

    for (i = 0; i < size; ++i)
        A[i][i] += shift;

    for (iter = 0; iter < max_iter; ++iter)
    {
        long double max_off = 0.0L;

        // dekompozycja (zmodyfikowany Gram-Schmidt)
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

        // złożenie (mnożenie w odwrotnej kolejności)
        for (i = 0; i < size; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                long double sum = 0.0L;
                // od 'i' bo R jest górno trójkątna
                for (int p = i; p < size; ++p)
                    sum += R[i][p] * Q[p][j];
                A[i][j] = sum;

                if (i != j && fabsl(sum) > max_off)
                {
                    max_off = fabsl(sum);
                }
            }
        }

        // macież diagonalna, koniec
        if (max_off < eps)
            break;
    }

    for (i = 0; i < size; ++i)
    {
        // odjęcie shifta
        values[i + 1] = A[i][i] - shift;
    }

    return 1;
}

// sprawdzene czy wartości własne są całkowite z dokładnością do 0.00001
int isintegral(int N, long double *x)
{
    int i;
    long double u;
    for (i = 1; i <= N; i++)
    {
        u = x[i];
        if (!((ceill(u) - u < 1e-5L) || (u - floorl(u) < 1e-5L)))
        {
            return 0;
        }
    }
    return 1;
}

// dokodowanie formatu graph6
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

    char BUFFOR[1024];
    long double x[17];
    matrix A;
    int N = 0;

    long num_checked = 0; // ilość sprawdzonych grafów
    long num_found = 0;   // ilość znalezionych grafów

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

    while (fgets(BUFFOR, 1024, in) != NULL)
    {
        num_checked++;
        BMKdecode(BUFFOR, &N, A);

        eigensymmatrix_qr(N, A, x);
        if (isintegral(N, x))
        {
            num_found++;
            fprintf(out, "%s", BUFFOR);
        }
    }

    printf("Sprawdzono %ld grafow\n", num_checked);
    printf("Znaleziono %ld grafow\n", num_found);

    fclose(out);
    fclose(in);

    return 0;
}