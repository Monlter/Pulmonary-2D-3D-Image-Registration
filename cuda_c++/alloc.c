/*****************************************************************************

    alloc.c - functions to create arrays, with syntax similar to calloc

    syntax:
        void *   alloc1d (int n,  int size);
        void **  alloc2d (int nx, int ny, int size);
        void *** alloc3d (int nx, int ny, int nz, int size);
        free1d(a);    ( macro )
        free2d(a);    ( macro )
        free3d(a);    ( macro )

        n, nx, ny, nz   size of 1D, 2D or 3D array
        returns         pointer to a 1D, 2D or 3D array
        a               pointer to array to be released

    by Guoping Han Dec. 3, 1997
    Rev 2.0 May 25, 1999

*****************************************************************************/

#include "alloc.h" 

//void * alloc1d (int  n, int  size)
//{		
//    void * a;
//    if ((a=malloc(n*size))==NULL)
//	fprintf(stderr, "warning: memory allocation failure.\n");
//    return a;
//}
//
//void ** alloc2d (int nx, int ny, int size)
//{
//    register int i, block;
//    void ** a;
//    a = (void **)alloc1d(ny, sizeof(void *));
//    a[0] = alloc1d(nx*ny, size);
//    block = nx*size;
//    for(i=1; i<ny; i++) a[i]=(void *)((int)(a[i-1])+block);
//    return a;
//}
//
//void *** alloc3d (int nx, int ny, int nz, int size)
//{
//    void *** a;
//    register int i, block;
//    a = (void ***) alloc2d(ny, nz, sizeof(void *));
//    a[0][0] = alloc1d(nx*ny*nz, size);
//    block = nx*size;
//    for(i=1; i<ny*nz; i++) a[0][i]=(void *)((int)(a[0][i-1])+block);
//    return a;
//}
float* alloc1d(int  n, int  size)
{
    float* a;
    if ((a = (float*)malloc(n * size)) == NULL)
        fprintf(stderr, "warning: memory allocation failure.\n");
    return a;
}

float** alloc2d(int nx, int ny, int size)
{
    register int i, block;
    float** a;
    a = (float**)alloc1d(ny, sizeof(float*));
    a[0] = alloc1d(nx * ny, size);
    block = nx * size;
    for (i = 1; i < ny; i++) a[i] = a[i - 1] + nx;
    return a;
}

float*** alloc3d(int nx, int ny, int nz, int size)
{
    float*** a;
    register int i, block;
    a = (float***)alloc2d(ny, nz, sizeof(float**));
    a[0][0] = alloc1d(nx * ny * nz, size);
    block = nx * size;
    for (i = 1; i < ny * nz; i++) a[0][i] = a[0][i - 1] + nx;
    return a;
}