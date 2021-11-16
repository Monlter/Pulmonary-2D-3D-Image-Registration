/*****************************************************************************

    frt.c - fast ray tracing algorithm

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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define EPS           1.E-5  /** 1.E-6 is too small **/
#define SWAP(x,y,dum) (dum)=(x), (x)=(y), (y)=(dum)

#ifdef fsqrt
#define SQRT(x) fsqrt(x)
#else
#define SQRT(x) sqrt((double)(x))
#endif

#ifdef ffloor
#define ROUNDDOWN(x) ( (int)(ffloor(x)) )
#define ROUNDUP(x)   ( (int)(fceil(x))  )
#else
#define ROUNDDOWN(x) ( (int)(floor(x)) )
#define ROUNDUP(x)   ( (int)(ceil(x))  )
#endif

#define  DEBUG(x)  /** printf(x) **/

/*--------------------------------------------------------------------------*/

int frt2d (int *nsize, float *from, float *to, int *vox, int *direc,
           int *vdim, float *length)
{
#define D 2
  int num;
  float totlen, lmin, lmax, xmin;
  float dx[D];
  int ux[D];
  double k[D], l[D], lambda;  /** to trace more accurately **/
  float dum0, dum1, dum;

  int i, d;


  /** setup **/

  for (i=0; i<D; i++)
  {
    dx[i] = to[i] - from[i];
    if (dx[i]>EPS)       direc[i] =  1;
    else if (dx[i]<-EPS) direc[i] = -1;
    else                 direc[i] =  0;
  }
  for (i=0, totlen=0.; i<D; i++) totlen += dx[i]*dx[i];
  totlen = SQRT(totlen);
  if (totlen <= 0.) return 0;

  DEBUG (("setup Ok\n"));


  /** initialize **/

  lmin = 0.;
  lmax = totlen;
  for (i=0; i<D; i++) 
  {
    if (direc[i])
    {
      k[i] = totlen/dx[i];
      dum0 = k[i] * (-from[i]);
      dum1 = k[i] * (nsize[i]-from[i]);
      if (dum0 > dum1) SWAP(dum0, dum1, dum);
      if (lmin<dum0) lmin = dum0;
      if (lmax>dum1) lmax = dum1;
    }
    else  /** bug removed by G Han 6/16/1999 **/
      if (from[i]<=0 || from[i]>=nsize[i]) return 0.;
  }
  if (lmin>=lmax) return 0;

  for (i=0; i<D; i++)
  {
    xmin = from[i] + lmin*dx[i]/totlen;
    vox[i] = ROUNDDOWN (xmin);
    if (xmin==vox[i] && direc[i]>0) vox[i]--;

    if (direc[i]>0) ux[i] = ROUNDUP (xmin);
    else ux[i] = ROUNDDOWN(xmin);

    if (direc[i]) l[i] = (ux[i]-from[i])*k[i];
    else          l[i] = lmax+lmax;  /* set as too large */
  }

  DEBUG (("initialization Ok\n"));


  /** ray-tracing: starting and trailing 0's not handled **/

  for (i=0; i<D; i++) if (k[i]<0) k[i]=-k[i];

  for (num=0, lambda=lmin; ; )
  {
    for (d=0, i=1; i<D; i++) if (l[i]<l[d]) d = i;

    if (l[d]>=lmax) /** last voxel **/ 
    {
      length[num] = lmax-lambda;
      vdim[++num] = d;
      break;
    }

    length[num] = l[d]-lambda;
    vdim[++num] = d;

    lambda = l[d];
    l[d]  += k[d];
  }

  DEBUG (("ray-tracing Ok\n"));

  return num;
#undef  D
}


/*--------------------------------------------------------------------------*/

int frt3d (int *nsize, float *from, float *to, int *vox, int *direc,
           int *vdim, float *length)
{
#define D 3
  int num;
  float totlen, lmin, lmax, xmin;
  float dx[D];
  int ux[D];
  double k[D], l[D], lambda;  /** to trace more accurately **/
  float dum0, dum1, dum;

  int i, d;


  /** setup **/

  for (i=0; i<D; i++)
  {
    dx[i] = to[i] - from[i];
    if (dx[i]>EPS)       direc[i] =  1;
    else if (dx[i]<-EPS) direc[i] = -1;
    else                 direc[i] =  0;
  }
  for (i=0, totlen=0.; i<D; i++) totlen += dx[i]*dx[i];
  totlen = SQRT(totlen);
  if (totlen <= 0.) return 0;

  DEBUG (("setup Ok\n"));


  /** initialize **/

  lmin = 0.;
  lmax = totlen;
  for (i=0; i<D; i++) 
  {
    if (direc[i])
    {
      k[i] = totlen/dx[i];
      dum0 = k[i] * (-from[i]);
      dum1 = k[i] * (nsize[i]-from[i]);
      if (dum0 > dum1) SWAP(dum0, dum1, dum);
      if (lmin<dum0) lmin = dum0;
      if (lmax>dum1) lmax = dum1;
    }
    else  /** bug removed by G Han 6/16/1999 **/
      if (from[i]<=0 || from[i]>=nsize[i]) return 0.;
  }
  if (lmin>=lmax) return 0;

  for (i=0; i<D; i++)
  {
    xmin = from[i] + lmin*dx[i]/totlen;
    vox[i] = ROUNDDOWN (xmin);
    if (xmin==vox[i] && direc[i]>0) vox[i]--;

    if (direc[i]>0) ux[i] = ROUNDUP (xmin);
    else ux[i] = ROUNDDOWN(xmin);

    if (direc[i]) l[i] = (ux[i]-from[i])*k[i];
    else          l[i] = lmax+lmax;  /* set as too large */
  }

  DEBUG (("initialization Ok\n"));


  /** ray-tracing: starting and trailing 0's not handled **/

  for (i=0; i<D; i++) if (k[i]<0) k[i]=-k[i];

  for (num=0, lambda=lmin; ; )
  {
    for (d=0, i=1; i<D; i++) if (l[i]<l[d]) d = i;

    if (l[d]>=lmax) /** last voxel **/ 
    {
      length[num] = lmax-lambda;
      vdim[++num] = d;
      break;
    }

    length[num] = l[d]-lambda;
    vdim[++num] = d;

    lambda = l[d];
    l[d]  += k[d];
  }

  DEBUG (("ray-tracing Ok\n"));

  return num;
#undef  D
}
