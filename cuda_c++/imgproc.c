/* imgproc.c by Hongbing Lu 04/17/02
 * Purpose: Procedures or subroutines used for image operation.
 * Copy right: Hongbing Lu and Zhengrong Liang, SUNY @ Stony Brook
 * Address: 
	Hongbing Lu, Ph.D.
	Department of Radiology
	State University of New York
	Stony Brook, NY 11974-8460
	hblu@hpjzl.rad.sunysb.edu
	Tel: 516-444-2508 Fax: 516-444-6450
 */

//#include <math.h>
#include <stdio.h>
#include "imgproc.h"

long int IMGREAD(filen, datap, size, header, dtype)
char *filen;
float *datap;
long int size, header;
int dtype;

{  FILE *fp;
   short *datas;
   double *datad;
   long int i, number;

  if((fp=fopen(filen,"rb"))==NULL) {
        printf("The file was not opened\n"); exit(1);
  }
  number=0;
  fseek(fp, header, 0);
  if (dtype==0){
    if ((datas=(short *)calloc(size,sizeof(short)))==NULL){
        printf("Memory allocation error\n");exit(1);}
    number+=fread(datas, sizeof(short),size, fp);
    for (i=0; i<size; i++)
        datap[i]= (float) datas[i];
    free(datas);
  }
  else
    if (dtype==2){
        if ((datad=(double *)calloc(size,sizeof(double)))==NULL){
          printf("Memory allocation error\n");exit(1);}
        number+=fread(datad, sizeof(double),size, fp);
        for (i=0; i<size; i++)
          datap[i]= (float) datad[i];
        free(datad);
    }
    else {
        number+=fread(datap, sizeof(float),size, fp);
    }

  if (fclose(fp))
        printf("The file was not closed\n");
  return number;
}


void IMGWRITE(filen, datap, size)
char *filen;
float *datap;
int size;

{
  FILE *fp;

  if((fp=fopen(filen, "wb"))==NULL) {
     printf("The file was not opened\n"); exit(1);
  }
  fwrite(datap, sizeof(float), size, fp);
  if (fclose(fp))
     printf("The file was not closed\n"); 
}
