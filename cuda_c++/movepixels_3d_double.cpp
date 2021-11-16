//#include "mex.h"

#include "math.h"
#include "image_interpolation.h"
#include "ap.h"
//#include "movepixels_3d_double.h"

/*   undef needed for LCC compiler  */
/*#undef EXTERN_C
#ifdef _WIN32
	#include <windows.h>
	#include <process.h>
#else
	#include <pthread.h>
#endif*/

/*  This function movepixels, will translate the pixels of an image
 *  according to x, y and z translation images (bilinear interpolated). 
 * 
 *  Iout = movepixels_3d_double(I,Tx,Ty,Tz);
 *
 *  Function is written by D.Kroon University of Twente (July 2009)
 */

//#ifdef _WIN32
  //unsigned __stdcall transformvolume(double **Args) {
//#else
void transformvolume(float *Iin, float *Iout, float *Txyz, int *Isize_d) 
  {
//#endif
    /* I is the input image, Iout the transformed image  */
    /* Tx and Ty images of the translation of every pixel. */
    //double *Iin, *Iout, *Tx, *Ty, *Tz;
    double *Nthreadsd;
    int Nthreads;
	/*  if one outside pixels are set to zero. */
	double  *moded;
	int mode=0;
   
    /* Cubic and outside black booleans */
    int black, cubic;
    
    /* 3D index storage*/
    int indexI;
    
    /* Size of input image */
   // double *Isize_d;
    int Isize[3]={0,0,0};
        
    /* Location of translated pixel */
    double Tlocalx;
    double Tlocaly;
    double Tlocalz;
    
    /* offset */
    int ThreadOffset=0;
    
    /* The thread ID number/name */
    double *ThreadID;
    
    /* X,Y coordinates of current pixel */
    int x,y,z;
	int xyzIndex,totalVoxel;
    

	Nthreads=1;


    cubic=0;
	black=0;

    Isize[0] = (int)Isize_d[0]; 
    Isize[1] = (int)Isize_d[1]; 
    Isize[2] = (int)Isize_d[2]; 

	totalVoxel = Isize[0]*Isize[1]*Isize[2];
       
    //ThreadOffset=(int) ThreadID[0];
	
    /*  Loop through all image pixel coordinates */
    for (z=ThreadOffset; z<Isize[2]; z=z+Nthreads)
	{
        for (y=0; y<Isize[1]; y++)
        {
            for (x=0; x<Isize[0]; x++)
            {

				xyzIndex=x+y*Isize[0]+z*Isize[0]*Isize[1];
				Tlocalx=((double)x)+Txyz[xyzIndex];
                Tlocaly=((double)y)+Txyz[xyzIndex+totalVoxel];
                Tlocalz=((double)z)+Txyz[xyzIndex+2*totalVoxel];

                
                /* Set the current pixel value */
                indexI=mindex3(x,y,z,Isize[0],Isize[1]);
                Iout[indexI]=interpolate_3d_double_gray(Tlocalx, Tlocaly, Tlocalz, Isize, Iin,cubic,black); 
            }
        }
    }

}

