#include<stdio.h>  
#include<time.h>  

int delay(int time)
{
    int i,j;
    for(i=0; i<time; i++)
        for(j=0; j<5000; j++)
            ;
}


int main()  
{
    struct timespec start = {0, 0};
    struct timespec end = {0, 0};
    double msdiff, usdiff;

    clock_gettime(CLOCK_REALTIME, &start);
    delay(10);
    clock_gettime(CLOCK_REALTIME, &end);
    msdiff = 1000.0*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1000000.0;
    usdiff = 1000000.0*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1000.0;
    printf("time passed is %lf ms\n", msdiff);
    printf("time passed is %lf us\n", usdiff);

    return 0;  
}
