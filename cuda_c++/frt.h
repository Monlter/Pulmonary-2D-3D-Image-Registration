/*****************************************************************************

    frt.h frt.c - fast ray tracing algorithm

    syntax:
    	int frt2d (int *nsize, float *from, float *to, int *vox, int *direc,
                   int *vdim, float *length)
    	int frt3d (int *nsize, float *from, float *to, int *vox, int *direc,
                   int *vdim, float *length)

	nsize      the size of the grid
	from, to   coordinates of the starting and end points
	vox        the indices of the first voxel on the ray (returned)
	direc      +/- 1, indicating increment/decrement in each direction 
                   or axis (returned)
	vdim       the axis to be incremented or decremented, with the first
                   undefined (returned)
	length     lengths of the ray in each voxel (returned)
	returns    the number of voxels on the ray inside the grid

    ** note **
        For efficiency reasons, the returned list of voxels may contain one
        or more voxels outside of the grid with 0 intersecting length.
        Segmentation fault can occur when those voxels are accessed. The
        calling function is expected to re-initialize the list to exclude
        access those voxels. Below is a sample to re-initialize and access
        the list of voxels:

           int i, d;
	   ...
	   for (i=0; i<num && length[i]<1.E-5; d=vdim[++i], vox[d]+=direc[d]) ;
	   for (; i<num && length[num-1]<1.E-5; num--) ;
	   for (; i<num; d=vdim[++i], vox[d]+=direc[d]) ...

        another way to access the list without re-initialization is:

           int i, d;
	   ...
           for (i=0; i<num; d=vdim[++i], vox[d]+=direc[d])
             if (length[i]>1.E-5) ...
        

    by Guoping Han, Rev 2.0 May 28, 1999

*****************************************************************************/

#ifndef FRT_H
#define FRT_H

extern int frt2d (int *nsize, float *from, float *to, int *vox, int *direc,
                  int *vdim, float *length);
extern int frt3d (int *nsize, float *from, float *to, int *vox, int *direc,
                  int *vdim, float *length);

#endif
