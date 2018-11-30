
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

int main()
{
    int i;
    double A[9] = { 1.0, 2.0, 1.0, -3.0, 4.0, 0.0, -1.0, 2.0, 3.0 };
    double B[9] = { 2.0, 3.0, 1.0, 1.0, -2.0, -3.0, -1.0, 3.0, 1.0 };
    double C[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double D[9] = { 3.0, 2.0, -4.0, -2.0, -17.0, -15.0, -3.0, 2.0, -4.0 };

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, A, 3, B, 3, 0, C, 3);
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
