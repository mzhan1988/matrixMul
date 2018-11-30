
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

int main()
{
    int i;
    double A[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    double B[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    double C[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
    double D[9] = { -5.0, 10.0, -1.0, 10.0, -10.0, 4.0, 7.0, 4.0, 5.0 };

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 2, 1, A, 2, B, 3, 0, C, 3);
    for (i = 0; i<9; i++)
    {
        printf("%lf ",C[i]);
    }
    printf("\n");
    for (i = 0; i<9; i++)
    {
        printf("%lf ",D[i]);
    }
    printf("\n");
    return 0;
}
