
#include <stdio.h>
#include <stdlib.h>

int main()
{
    //srand(2018);
    printf("%d\n", RAND_MAX);
    int i;
    double value;
    for(i=0; i<10; ++i)
    {
        printf("%d\n", rand());
    }
    
    for(i=0; i<10; ++i)
    {
        value = rand() / (double)RAND_MAX;
        printf("%lf\n", value);
    }
    return 0;
}
