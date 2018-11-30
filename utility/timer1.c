#include<stdio.h>  
#include<sys/time.h>  

int delay(int time)
{
    int i,j;
    for(i=0; i<time; i++)
        for(j=0; j<5000; j++)
            ;
}


int main()  
{
    struct timeval start;
    struct timeval end;
    double msdiff, usdiff;

    gettimeofday(&start, NULL);
    delay(10);
    gettimeofday(&end, NULL);
    msdiff = 1000.0*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000.0;
    usdiff = 1000000.0*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1.0;
    printf("time passed is %lf ms\n", msdiff);
    printf("time passed is %lf us\n", usdiff);

    return 0;  
}
