/*******************************************************************************

    orbitz.c - z axis value of source for dfferent scanning orbits.

    syntax:
        float orbitz(int nize, int zdiff, int nview, int view, int ftype)

        nsize       the number of object slices
	zdiff	  start difference between source and object along z axis,
		  in pixel size
	nview	  number of projection views
	view	  current projection view
	ftype	  type of scanning orbits, can be 1--circular, 2--spiral, 
		  3--2-half-cir, 4--2-half-spiral, 5--half-circular+spiral

    by Hongbing Lu, Rev Sep 20, 2002.

*******************************************************************************/

#include <stdio.h>
//#include <math.h>
#define PI M_PI

float orbitz(int nsize, int zdiff, int nview, int view, int ftype)

{
  float zz;
  switch (ftype)
  {
    case 1:
	zz = (float)zdiff-nsize/2.; 
	break;

    case 2:
	zz = view*nsize/nview-nsize/2.;
	break;

    case 3:
	if (view<nview/2) zz = -nsize/2.;
        else zz = nsize/2.;
	break;

    case 4:
	if (view <nview/2) zz = 2.*view*nsize/nview-nsize/2.;
        else zz = 2.*view*nsize/nview-3.*nsize/2.;
	break;

    case 5:
	if (view<nview/2) zz = -nsize/2.;
	else zz = 2.*view*nsize/nview-3.*nsize/2.;
	break;

    default: printf("Invalid orbit type, exit...\n"); exit(0);
  }
  return (zz);
}

