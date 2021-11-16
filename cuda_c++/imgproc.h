/**************** imgproc.h ******************************/

static long int IMGREAD(char *filen, float *datap,long int size, long int header, int dtype);
static void IMGWRITE(char *filen,float *datap, int size);
//void IMGWRITE(char *filen, float** datap,int nx, int ny);
