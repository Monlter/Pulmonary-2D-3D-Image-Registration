#include <stdlib.h>
#include <stdio.h>

#define free1d(a)  free(a)
#define free2d(a)  free(a[0]), free(a)
#define free3d(a)  free(a[0][0]), free2d(a[0])

