#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include "getopt.h"
#include "alloc.h"
#include "alloc.c"
#include "imgproc.cpp"
#include <math.h>
#include <time.h>
#include "orbitz.c"
#include <string.h>


// includes, cuda
#include <helper_cuda.h>
#include <helper_timer.h>
#include <device_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cuda.h>
#include <cublas.h>
#include <helper_math.h>
#include <helper_functions.h>
//自己定义
#include "read_file.h"



// here the order of the "image_interpolation.h", "image_interpolation.cpp" and "movepixels_3d_double.cpp" is  very important. Otherwise error will happen!!
#include "image_interpolation.h"
//#include "image_interpolation.cpp"
//#include "movepixels_3d_double.cpp"



#define EPS 1.0E-4
#define M_PI 3.1415926535
//#define NI_X 512
//#define NI_Z 384
//
//#define SID 1500   // in mm
//#define SOD 1000  // in mm

//-------------------------------------------------------------------------------------------------------------------------------
//#define L1 1000
////      distance from isocenter to source,in mm
//#define L2 500
////      distance from isocenter to imager, in mm
//#define VOXSIZE_X 2.0 
//#define VOXSIZE_Z 2.0
//	the finest scale voxel size, in unit of mm
#define TOL 1e-4
//      the tolarence used in ray tracing, small TOL results in projection error
#define EPS_xunjia 1e-8


#define PIXSIZE_X 1
#define PIXSIZE_Z 1
//      pixel size in unit of mm
//------------------------------------------------------------------------------------------------------------------------------

// CUDA parameters
#define NBLOCKX 32768
//      the leading dimension of the 2D thread grid
#define NTHREAD_PER_BLOCK 256

#define DEVICENUMBER 3
#define PIXELSIZE 1  // in mm

int deviceCount;
cudaDeviceProp dP;


# define EPS 1.0E-4
# define M_PI 3.1415926535
# define NTHREAD_PER_BLOCK 128
//-------------------------------------------------------------------------------------------------------------------------------

//--------------below is the #defined parameters used for frt3d function--------------------------//

#define EPS_frd3d           1.E-6  /** 1.E-6 is too small **/
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


// functions declaration 
float orbitz(int nsize, int zdiff, int nview, int view, int ftype);

__device__ int frt3d_kernel(int* SIZE, float* phantom, float* From, float* To, int* Vox, int* Direc, int* Vdim, float* Length, int nx, int nz);
__global__ void forwardProj2d(float* dest_projection, float* phantom, int nx, int nz, float vx, float vz, float sine, float cosin, float SID, float SOD, int NI_X, int NI_Z);
__device__ float forwardProjRay(float* phantom, float alphaMin, float alphaMax, int startDim, int direction[3], float dAlpha[3], float length, float xi, float yi, float zi, float xs, float ys, float zs,
	int nx, int nz, float vx, float vz);
__global__ void backwardProj2d(float* dest, float* proj_diff2d, int currentProjIdx, int TotalProjNum, int nx, int nz, float vx, float vz, float sine_value, float cosin_value);

__global__ void backwardProj3d(float* dest, float* d_prj3d, int NPRJ, int nx, int nz, float vx, float vz, float* sine, float* cosin);

__host__ __device__ int ind3to1(int i, int j, int k, int ni, int nj, int nk)
//      convert a 3-component real index into a single index
{
	return i + j * ni + k * ni * nj;
}

__host__ __device__ void ind1to3(int id, int ni, int nj, int nk, int index[3])
//      convert a voxex sigle index into real indices
{
	index[0] = id % ni;
	index[2] = id / (ni * nj);
	index[1] = (id - index[0] - index[2] * ni * nj) / ni;

	return;
}

class CBCTGeo {
public:
	int nx, ny, nz;
	int dx, dy, nview;
	float sdd, sod;
	float voxelsize, dbinsize;
};


float sumsq(float* x, int num);

///////////////////////////////------funcgrad-------------------/////////////////
__global__ void DVF_ajust(float* d_x, int offset, int voxelNumber);
__global__ void funcgrad_cu(int dim, int nx, int nz, float weight, float* h_voxelgrad, float* d_x, float* d_grad, float* d_warps, float* d_warpsM, float* d_energysum);

__global__ void w_genenator(int dim, int nx, int nz, float* d_boundary_norm, float* d_w);

__global__ void boundary_norm_expand(int dim, int nx, int nz, float* d_boundary_norm, float* d_boundary_expand);

__global__ void funcgrad_sliding_cu3_step1(int dim, int nx, int nz, float weight, float* h_voxelgrad, float* d_x, float* d_grad, float* d_warps, float* d_warpsM, float* d_energysum, float* d_boundary_norm, float* d_product1_0, float* d_product1_1, float* d_product1_2, float* d_product2_0, float* d_product2_1, float* d_product2_2,
	float* d_test1, float* d_test2);

__global__ void DVF_inverse(float* d_x, int size);

__global__ void funcgrad_sliding_cu3_step2(int dim, int nx, int nz, float weight, float* d_voxelgrad, float* d_x3d, float* d_grad, float* d_warps, float* d_warpsM, float* d_energysum,
	float* d_boundary_norm, float* d_product1_0, float* d_product1_1, float* d_product1_2, float* d_product2_0, float* d_product2_1, float* d_product2_2, float* d_test2);

__global__ void funcgrad_sliding_cu4_step1(int dim, int nx, int nz, float weight, float* d_voxelgrad, float* d_x3d, float* d_grad, float* d_warps, float* d_warpsM, float* d_energysum,
	float* d_boundary_norm, float* d_product1_0, float* d_product1_1, float* d_product1_2, float* d_product2_0, float* d_product2_1, float* d_product2_2,
	float* d_test1, float* d_test2);

//__global__ void funcgrad_sliding2_cu3_step1(int dim, int nx, int nz, float weight, float *h_voxelgrad, float *d_x, float *d_grad, float *d_warps, float *d_warpsM, float *d_energysum,float *d_boundary_norm, float *d_product1_0,  float *d_product1_1, float *d_product1_2, float *d_product2_0,  float *d_product2_1, float *d_product2_2,
//							 float *d_test1, float *d_test2,float *d_w);

__global__ void funcgrad_sliding2_cu3_step1(int dim, int nx, int nz, float weight, float* d_voxelgrad, float* d_x3d, float* d_grad, float* d_warps, float* d_warpsM, float* d_energysum,
	float* d_boundary_norm, float* d_product1_0, float* d_product1_1, float* d_product1_2, float* d_product2_0, float* d_product2_1, float* d_product2_2,
	float* d_test1, float* d_test2, float* d_w, float* d_boundary_norm_expand);



__global__ void GradSum(float* d_grad1, float* d_grad2, int size);
__global__ void DVF_update(float* d_cong, float* d_x, int t, int size);



__global__ void Get_condg(float* d_grad, float* d_cong, float* d_condg, int size);

__global__ void cong_update(float* d_cong, float* d_grad, float beta, int size);

__global__ void Get_gradold(float* d_grad, float* d_gradold, int size);

__global__ void Get_deltanew(float* d_deltanew, float* d_grad, float* d_gradold, int size);


//void transformvolumeGPU(float *Iin, float *Iout, float *Txyz, float *d_Txyz, int *Isize_d);
void transformvolumeGPU(float* Iin, float* Iout, float* Txyz, int* Isize_d);

__global__ void interpolate_3d_double_gray_cu(float* d_Iout, float* d_Iin, float* d_Txyz, int* d_Isize, int cubic, int black);

__device__ double interpolate_3d_double_gray_core_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* d_Isize, float* Iin, int cubic, int black);
__device__ double interpolate_3d_cubic_black_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* d_Isize, float* Iin);
__device__ double interpolate_3d_linear_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* d_Isize, float* Iin);

__device__ double interpolate_3d_cubic_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* d_Isize, float* Iin);
__device__ double interpolate_3d_linear_black_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* d_Isize, float* Iin);

__device__ double pow2_cu(double val);
__device__ double pow3_cu(double val);
__device__ double getcolor_mindex3_cu(int x, int y, int z, int sizx, int sizy, int sizz, float* I);


float funcgrad_NoSegmentation_CPU(int dim, int nx, int nz, float lamda, float* h_voxelgrad, float* x3d, float* h_grad, float*** warps, float*** warpsM, float h_energysum, int w, float*** spacial_w, float sigma_d, float sigma_r, float sigma_v);
float funcgrad_NoSegmentation_CPU2(int dim, int nx, int nz, float lamda, float* h_voxelgrad, float* x3d, float* h_grad, float*** warps, float*** warpsM, float h_energysum, int w, float sigma_d, float sigma_r, float sigma_v);
__global__ void funcgrad_cu_Bilateral(int dim, int nx, int nz, float weight, float* d_voxelgrad, float* d_x3d, float* d_grad, float* d_warps, float* d_warpsM, float* d_energysum, float sigma_d, float sigma_r, float sigma_v);
__global__ void	funcgrad_NoSegmentation(int dim, int nx, int nz, float weight, float* d_voxelgrad, float* d_x3d, float* d_grad, float* d_warps, float* d_warpsM, float* d_energysum, int w, float sigma_d, float sigma_r, float sigma_v, float* d_ss);


float abso(float a);
float maxi(float a, float b);
float mini(float a, float b);

// start program..
static void usage(char* cmd)
{
	printf("%s: Simulation of cone-beam projection for circular or spiral scanning orbits.\n", cmd);
	printf("\nSyntax: %s [options] input.img proj.img\n\n\
        input.img:      filename of the input object image\n\
        proj.img:       filename of output projection image\n\n\
        Options available:\n\
        [-size lxmxz]   size of input image, default 128x128x64\n\
        [-header h]     for header length in input image, default 0 \n\
        [-dsize bxsxn]  detector size and number of views, format (bin)x(height)x(view)\n\
        [-geo SID SOD]  source-to-image(detector) distance and source-to-object distance, default 300 200\n\
	    [-pos sz dz]	start source and detector position along the rotation axis z(relative to object), in pixel size\n\
	    [-orbit type]	scanning orbits, 1--circular(default), 2--spiral, 3--2-half-cir, 4--2-half-spiral,\n\
			5--half circular+spiral\n\
	    [-180]		180-degree scanning\n\n", cmd);
	exit(-1);
}
int argc;
char* argv[];




int main(int argc, char* argv[])
{
	printf("beig");

	//*************变量声明*******************************
	float*** projection, *** temp, *** h_proj, *** warps, *** CT_deform;
	float* h_prj2d, * d_temp, * d_prj3d, * d_prj2d, * CT_ref, * x3d, * d_x3d, * x3d_moving, * h_sine, * h_cosin, * h_angle;
	char* cmd, * projfile1, * anglefile1, * priorimage1, * outfile1;
	int size[3], dsize[3],dsize2[3], pos[2];
	float voxelsize, binsize, SID, SOD, L1, L2, VOXSIZE_X, VOXSIZE_Z;
	int m, DIM, nx, nz, nview, VoxelNum, NPRJ, nview1,nview2, ITNUM, flag, orbitype, i, j, k, n, ii, jj, kk, NI_X, NI_Z;
	char str_angle_file[100],str_number[3], str_infolder[200], str_outfolder[200], 
		dvf_path[500],ct_path[500],model_class[20],dvf_list[20],str_infilename[60], 
		str_outfilename[60], str_infile[200], str_ct_outfile[500], str_projection_outfile[500],
		DVF_name[300],CT_name[300],Projection_name[300];
	int datasetr, datasetp;
	bool save_CT_flag,save_Projection_flag;




	cmd = argv[0];
	//while ((--argc > 0) && ((*++argv)[0] == '-'))
	//{
	//	/*std::cout << argv[0]+1 << std::endl;*/
	//	if (!strcmp(argv[0] + 1, "size")) GETOPTSIZ(size)   //size :256*256*150
	//	else if (!strcmp(argv[0] + 1, "view")) GETOPTINT(nview1)  //nview1:100
	//	else if (!strcmp(argv[0] + 1, "voxelsize")) GETOPTFLT(voxelsize)  //voxelsize:1.0
	//	else if (!strcmp(argv[0] + 1, "itnum")) GETOPTINT(ITNUM)   //itnum: 30

	//	else if (!strcmp(argv[0] + 1, "dsize")) GETOPTSIZ(dsize)  //dsize:300*240*100
	//	else if (!strcmp(argv[0] + 1, "geo"))
	//	{
	//		SID = atof(argv[1]);  //sid :1500
	//		argc--; argv++;
	//		SOD = atof(argv[1]);   //sod:1000
	//		argc--; argv++;
	//	}
	//	else usage(cmd);
	//}

	//if (size[0] != size[1])
	//{
	//	fprintf(stderr, "%s warning: image must be square\n", cmd);
	//	size[1] = size[0];
	//}

	//if (argc != 1)
	//	usage(cmd);
	//anglefile1 = argv[0];  


	// <*************************************************************************************************************************>
    // 参数 ----> 输入
	size[0] = 256;
	size[1] = 256;
	size[2] = 150;
	nview1 = 1;
	voxelsize = 1.0;
	ITNUM = 30;
	dsize[0] = 300;
	dsize[1] = 240;
	dsize[2] = 1;
	SID = 750;
	SOD = 500;
	char anglefile[] = "new_anglefile_1_angle";
	char DVF_path_list[] = "D:\\Code\\Pycharm\\Pulmonary-2D-3D-Image-Registration\\Out_result\\spaceAndTime_PCA\\DVF_path.txt";
	char CT_path_list[] = "D:\\Code\\Pycharm\\Pulmonary-2D-3D-Image-Registration\\Out_result\\spaceAndTime_PCA\\CT_path.txt";
	char Projection_path_list[] = "";
	char reference_ct_path[] = "D:\\Code\\Pycharm\\Pulmonary-2D-3D-Image-Registration\\Out_result\\spaceAndTime_PCA\\phantom_CT_0.bin";
	save_CT_flag = true;
	save_Projection_flag = false;
	// 参数 ----> 结束
	// 转变方式：
	// 1. DVF -> CT -> projection (批量)
	// 2. CT -> projection (批量)
	// 3. DVF -> CT -> projection (单个)
	// 4. CT -> projection (单个)
	// <*************************************************************************************************************************>



	int voxelNumber = size[0] * size[1] * size[2];
	int maxits = ITNUM; //30
	float relax = 0.15;
	datasetr = size[0] * size[1] * size[2];     //CT数据的尺寸
	datasetp = dsize[2] * dsize[0] * dsize[1];   //投影数据的尺寸
	m = 1;
	nview = dsize[2];
	DIM = 3;
	nx = size[0];
	nz = size[2];
	NI_X = dsize[0];
	NI_Z = dsize[1];
	VOXSIZE_X = voxelsize;
	VOXSIZE_Z = voxelsize;

	cublasStatus status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		getchar();
		exit(0);
	}
	//initialize CUBLAS

	printf("Object prior image_1 size: %dx%dx%d\n", size[0], size[1], size[2]);
	printf("Projection data dsize: %d x %d x %d\n", dsize[0], dsize[1], dsize[2]);
	printf("view1 = %d, voxelsize = %f, itnum = %d  \n", nview1, voxelsize, ITNUM);
	printf("SID = %f,   SOD = %f\n", SID, SOD);

	/*************************************************************************************************************************
	*********************************正式程序从这里开始******************************************************************************
	************************************************************************************************************************/


	//************申请地址******************************

	warps = (float***)alloc3d(size[0], size[1], size[2], sizeof(float));

	x3d = (float*)alloc1d(3 * voxelNumber, sizeof(float));

	// new added angle....
	h_angle = (float*)alloc1d(dsize[2], sizeof(float));

	h_sine = (float*)alloc1d(dsize[2], sizeof(float));      // dsize[2]个投影

	h_cosin = (float*)alloc1d(dsize[2], sizeof(float));


	CT_deform = (float***)alloc3d(size[0], size[1], size[2] * m, sizeof(float));

	temp = (float***)alloc3d(size[0], size[1], size[2], sizeof(float));

	x3d_moving = (float*)alloc1d(3 * voxelNumber * m, sizeof(float));

	CT_ref = (float*)alloc1d(voxelNumber, sizeof(float));

	// allocate host 2d projection memory
	h_prj2d = (float*)alloc1d(dsize[0] * dsize[1], sizeof(float));
	// allocate host 3d projection memory
	h_proj = (float***)alloc3d(dsize[0], dsize[1], dsize[2], sizeof(float));
	projection = (float***)alloc3d(dsize[0], dsize[1], dsize[2], sizeof(float));

	checkCudaErrors(cudaMalloc((void**)&d_x3d, sizeof(float) * 3 * voxelNumber));
	checkCudaErrors(cudaMalloc((void**)&d_temp, sizeof(float) * size[0] * size[1] * size[2]));   // 申请 d_temp（3） 内存空间
	checkCudaErrors(cudaMemset(d_temp, 0, sizeof(float) * size[0] * size[1] * size[2]));
	checkCudaErrors(cudaMalloc((void**)&d_prj2d, sizeof(float) * dsize[0] * dsize[1]));
	checkCudaErrors(cudaMemset(d_prj2d, 0, sizeof(float) * dsize[0] * dsize[1]));
	checkCudaErrors(cudaMalloc((void**)&d_prj3d, sizeof(float) * dsize[0] * dsize[1] * dsize[2]));
	checkCudaErrors(cudaMemset(d_prj3d, 0, sizeof(float) * dsize[0] * dsize[1] * dsize[2]));


	// cuda settings
	int N = dsize[0] * dsize[1];  // N:投影图像的尺寸
	dim3 nblocks;
	nblocks.x = NBLOCKX;                                                 // NBLOCKX:32768
	nblocks.y = ((1 + (N - 1) / NTHREAD_PER_BLOCK) - 1) / NBLOCKX + 1;   //NTHREAD_PER_BLOCK:128
	NPRJ = dsize[2];


	printf("size[2] = %d \n", size[2]);
	printf("size[1] = %d \n", size[1]);
	printf("size[0] = %d \n", size[0]);


	//************ 变量初始化 **********************************************
	for (i = 0; i < voxelNumber * 3; i++)
		x3d[i] = 0.0;

	// CT_deform 储存的是所有变形后的CT, temp 用于储存单个变形后的CT
	for (k = 0; k < size[2] * m; k++)
		for (j = 0; j < size[1]; j++)
			for (i = 0; i < size[0]; i++)
				CT_deform[k][j][i] = 0.0;


	for (k = 0; k < size[2]; k++)
		for (j = 0; j < size[1]; j++)
			for (i = 0; i < size[0]; i++)
			{
				temp[k][j][i] = 0.0;
				warps[k][j][i] = 0.0;
			}


	
	// projection 用于储存所有的投影
	for (k = 0; k < dsize[2]; k++)
		for (j = 0; j < dsize[1]; j++)
			for (i = 0; i < dsize[0]; i++)
			{
				projection[k][j][i] = 1.0;
				h_proj[k][j][i] = 0.0;
			}

	/*for (i = 0; i < datasetp; i++)
		h_proj[0][0][i] = 0.0;*/

	float sin_value = 0.0;
	float cos_value = 0.0;





	// input parameters for kernel function: forwardProj
	VoxelNum = size[2] * size[1] * size[0];

	double  timing;
	time_t  start, end;
	start = time(NULL);

	/*cudaMemcpy(d_x3d, x3d, sizeof(float) * voxelNumber * 3, cudaMemcpyHostToDevice);*/






//******** 1. 批量生成: DVF->CT->projection  ***************************************************************************************
	IMGREAD(reference_ct_path, CT_ref, voxelNumber, 0, 1); //读取参考相位的CT
	std::cout << "加载CT_ref" << std::endl;
	IMGREAD(anglefile, h_angle, dsize[2], 0, 1);   //  加载投影角度---> h_angle
	std::cout << "加载angle" << std::endl;
	for (i = 0; i < dsize[2]; i++)
	{
		h_angle[i] = h_angle[i] * M_PI / 180;
		h_sine[i] = sin(h_angle[i] - M_PI);
		h_cosin[i] = cos(h_angle[i] + M_PI);
	}

	//读取文件名
	FILE* DVF_list;
	FILE* CT_list;
	FILE* Projection_list;
	if (fopen_s(&DVF_list, DVF_path_list, "r") != 0)
		std::cout << "加载DVF_path_list.txt失败" << std::endl;
	if (save_CT_flag) {
		if (fopen_s(&CT_list, CT_path_list, "r") != 0)
			std::cout << "加载CT_path_list.txt失败" << std::endl;
	}
	if (save_Projection_flag) {
		if (fopen_s(&Projection_list, Projection_path_list, "r") != 0)
			std::cout << "加载Projection_path_list.txt失败" << std::endl;
	}
	
	while (!feof(DVF_list))
	{
		//获取文件夹下的所有文件
		fgets(DVF_name, sizeof(DVF_name), DVF_list); 
		//去除最后的换行符
		DVF_name[strlen(DVF_name) - 1] = 0;
		std::cout << DVF_name << "已经加载" << std::endl;
		  
		
	


		// *******生成CT操作*******************************************************
		IMGREAD(DVF_name, x3d_moving, 3 * voxelNumber, 0, 1);  //读取dvf
		transformvolumeGPU(CT_ref, warps[0][0],x3d_moving,size);
		//保存转换后的ct
		if (save_CT_flag) {
			fgets(CT_name, sizeof(CT_name), CT_list);
			CT_name[strlen(CT_name) - 1] = 0;
			IMGWRITE(CT_name, warps[0][0], voxelNumber);
			std::cout << CT_name << "已经保存" << std::endl;
		}
		//************进行投影******************************
		if (save_Projection_flag) {
			fgets(Projection_name, sizeof(Projection_name), Projection_list);
			Projection_name[strlen(Projection_name) - 1] = 0;
			cudaMemcpy(d_temp, warps[0][0], sizeof(float) * size[0] * size[1] * size[2], cudaMemcpyHostToDevice);   // 将 temp（3） 放在GPU中
			cudaMemcpy(temp[0][0], d_temp, sizeof(float) * size[0] * size[1] * size[2], cudaMemcpyDeviceToHost);
			for (n = 0; n < nview; n++)  // nview:100
			{
				sin_value = h_sine[n];
				cos_value = h_cosin[n];
				forwardProj2d <<<nblocks, NTHREAD_PER_BLOCK >>> (d_prj2d, d_temp, nx, nz, VOXSIZE_X, VOXSIZE_Z, sin_value, cos_value, SID, SOD, NI_X, NI_Z);  //投影d_temp（3）到 d_prj2d(2)
				cudaThreadSynchronize();   //该方法将停止CPU端线程的执行，直到GPU端完成之前CUDA的任务
				cudaMemcpy(&d_prj3d[n * dsize[0] * dsize[1]], d_prj2d, sizeof(float) * dsize[0] * dsize[1], cudaMemcpyDeviceToDevice); //将每个投影d_prj2d(2)放在d_prj3d(3)中
			}
			cudaMemcpy(h_proj[0][0], d_prj3d, sizeof(float) * dsize[0] * dsize[1] * dsize[2], cudaMemcpyDeviceToHost);  // 将d_prj3d(3) 复制到 h_proj(3)中
			IMGWRITE(Projection_name, h_proj[0][0], datasetp);
			std::cout << Projection_name << "投影已经生成" << std::endl;
		}
		
	}
	fclose(DVF_list);
	if (save_CT_flag)
		fclose(CT_list);
	if(save_Projection_flag)
		fclose(Projection_list);







// *************** 2.单独生成CT ***********************************************
	////读入一列DVF,其内含100个DVF，并依次提取单个DVF，用于变形参考相位图像；
	//std::cout << "读取dvf" << std::endl;
	//IMGREAD("G:\\Monlter\\PCA\\Pulmonary-2D-3D-Image-Registration\\Dataset\\Test_9dvf\\Output\\dvf\\Resnet(origin_MSE)\\predict_dvf_1", x3d_moving, 3 * voxelNumber, 0, 1);    // 加载m个dvf，保存在x3d_moving 中
	//// 读入参考相位CT图像
	//std::cout << "读取参考相位CT" << std::endl;
	//IMGREAD("4d_lung_phantom_w_lesion_atn_1.bin ", CT_ref, voxelNumber, 0, 1);
	//std::cout << "我加载CT和dvf" << std::endl;
	//// DVF + CT_ref进行CT的变形，并保存在warp中
	//transformvolumeGPU(CT_ref, warps[0][0], x3d_moving, size);
	//IMGWRITE("G:\\Monlter\\PCA\\Pulmonary-2D-3D-Image-Registration\\Dataset\\Test_9dvf\\Output\\CT\\predict_ct_1", warps[0][0], voxelNumber);
	//std::cout << "here is ok...test2" << std::endl;


//*************** 2.2 批量DVF转CT ***********************************************
	//读入一列DVF,其内含100个DVF，并依次提取单个DVF，用于变形参考相位图像；
	//读取文件名
	//FILE* fp1;
	//if (fopen_s(&fp1, "G:\\Monlter\\PCA\\Pulmonary-2D-3D-Image-Registration\\Dataset\\Test_9dvf\\Output\\input_list.txt", "r") != 0)
	//	std::cout << "加载model_class.txt失败" <<std::endl;
	//FILE* fp2;
	//if (fopen_s(&fp2, "G:\\Monlter\\PCA\\Pulmonary-2D-3D-Image-Registration\\Dataset\\Test_9dvf\\Output\\out_list.txt", "r") != 0)
	//	std::cout << "加载list.txt失败" << std::endl;
	//while (!feof(fp1))
	//{
	//	fgets(str_infile, sizeof(str_infile), fp1); //获取文件夹下的所有文件
	//	str_infile[strlen(str_infile) - 1] = 0;  //去除最后的换行符
	//	fgets(str_ct_outfile, sizeof(str_ct_outfile), fp2); //获取文件夹下的所有文件
	//	str_ct_outfile[strlen(str_ct_outfile) - 1] = 0;  //去除最后的换行符
	//		
	//	//读取文件
	//	IMGREAD(str_infile, x3d_moving, 3 * voxelNumber, 0, 1);    // 加载m个dvf，保存在x3d_moving 中
	//	std::cout << str_infile + strlen("G:\\Monlter\\PCA\\Pulmonary-2D-3D-Image-Registration\\Dataset\\Test_9dvf\\Output\\dvf\\") << ":\tdvf已加载\t";
	//	// 读入参考相位CT图像
	//	IMGREAD("4d_lung_phantom_w_lesion_atn_1.bin ", CT_ref, voxelNumber, 0, 1);
	//	// DVF + CT_ref进行CT的变形，并保存在warp中
	//	transformvolumeGPU(CT_ref, warps[0][0], x3d_moving, size);
	//	IMGWRITE(str_ct_outfile, warps[0][0], voxelNumber);
	//	std::cout << "ct已经保存" << std::endl;

	//}
	//std::cout << "here is ok...test1" << std::endl;

	

// ************ 3. 单独生成投影 ****************************************************

	////读取投影角度
	//IMGREAD("new_anglefile_100angle", h_angle, dsize[2], 0, 1);   //  加载投影角度---> h_angle
	//for (i = 0; i < dsize[2]; i++)
	//{
	//	h_angle[i] = h_angle[i] * M_PI / 180;
	//	h_sine[i] = sin(h_angle[i] - M_PI);
	//	h_cosin[i] = cos(h_angle[i] + M_PI);
	//}
	////读取转换后的CT图像
	//IMGREAD("G:\\Monlter\\PCA\\Dataset\\CT\\4d_lung_phantom_w_lesion_atn_1_shape256_256_150.bin", temp[0][0], voxelNumber, 0, 1);
	//// forward projection...
	//cudaMemcpy(d_temp, temp[0][0], sizeof(float) * size[0] * size[1] * size[2], cudaMemcpyHostToDevice);   // 将 temp（3） 放在GPU中
	//cudaMemcpy(d_prj3d, projection[0][0], sizeof(float) * dsize[0] * dsize[1] * nview, cudaMemcpyHostToDevice);  // 将projection（3） 放在GPU中
	//////////////////////////////////////////
	//// 2D projection each view...
	/////////////////////////////////////////
	//for (n = 0; n < nview; n++)  // nview:100
	//{
	//	sin_value = h_sine[n];
	//	cos_value = h_cosin[n];
	//	//printf("sin_value = %f \n",h_sine[n]);
	//	forwardProj2d << <nblocks, NTHREAD_PER_BLOCK >> > (d_prj2d, d_temp, nx, nz, VOXSIZE_X, VOXSIZE_Z, sin_value, cos_value);
	//	cudaThreadSynchronize();
	//	cudaMemcpy(&d_prj3d[n * dsize[0] * dsize[1]], d_prj2d, sizeof(float) * dsize[0] * dsize[1], cudaMemcpyDeviceToDevice); //将每个投影d_prj2d(2)放在d_prj3d(3)中
	//}
	//cudaMemcpy(h_proj[0][0], d_prj3d, sizeof(float) * dsize[0] * dsize[1] * dsize[2], cudaMemcpyDeviceToHost);  // 将d_prj3d(3) 复制到 h_proj(3)中
	//IMGWRITE("G:\\Monlter\\PCA\\Dataset\\other\\projection_0_phase", h_proj[0][0], datasetp);
	//printf("The simulated projection has been saved in: %s\n", str_number);






















	////读入一列DVF,其内含100个DVF，并依次提取单个DVF，用于变形参考相位图像；
	//IMGREAD("E:\\code\\pycharm\\CNN_PCA\\Dataset\\val_gen\\dvf_prediction_0_0", x3d_moving, 3 * voxelNumber * m, 0, 1);    // 加载m个dvf，保存在x3d_moving 中
	//// 读入参考相位CT图像
	//IMGREAD("4d_lung_phantom_w_lesion_atn_1.bin ", CT_ref, voxelNumber, 0, 1);
	//std::cout << "我加载CT和dvf" << std::endl;
	//for (j = 0; j < m; j++)
	//{
	//	for (k = 0; k < 3 * voxelNumber; k++)
	//		x3d[k] = x3d_moving[j * 3 * voxelNumber + k];   // x3d[] 保存每个DVF数值
	//	// DVF + CT_ref进行CT的变形，并保存在warp中
	//	transformvolumeGPU(CT_ref, warps[0][0], x3d, size);    
	//	// 将所有变形后的数据  全部保存在CT_deform
	//	for (kk = 0; kk < size[2]; kk++)
	//		for (jj = 0; jj < size[1]; jj++)
	//			for (ii = 0; ii < size[0]; ii++)
	//				CT_deform[j * size[2] + kk][jj][ii] = warps[kk][jj][ii];   	
	//}
	//IMGWRITE("E:\\code\\pycharm\\CNN_PCA\\Dataset\\val_gen\\CT_prediction_0_0", warps[0][0], voxelNumber * m);
	//std::cout << "here is ok...test2" << std::endl;







	// *******************  2.根据CT进行投影   生成DRR   *******************************************************************************************
	////读取投影角度
	//IMGREAD("new_anglefile_100angle", h_angle, dsize[2], 0, 1);   //  加载投影角度---> h_angle
	//for (i = 0; i < dsize[2]; i++)
	//{
	//	h_angle[i] = h_angle[i] * M_PI / 180;
	//	h_sine[i] = sin(h_angle[i] - M_PI);
	//	h_cosin[i] = cos(h_angle[i] + M_PI);
	//}
	// 读取转换后的CT图像


	//IMGREAD("D:/Dateset/dang/CT_gen/CT_0_0", CT_deform[0][0], VoxelNum * m, 0, 1);

	//for (i = 0; i < m; i++)
	//{
	//	// 将CT_deform中的CT复制到temp中
	//	for (kk = 0; kk < size[2]; kk++)
	//		for(jj = 0;jj < size[1]; jj++)
	//			for (ii = 0; ii < size[0]; ii++)
	//				temp[kk][jj][ii] = CT_deform[i * size[2] + kk][jj][ii];
	//			
	//	// forward projection...
	//	cudaMemcpy(d_temp, temp[0][0], sizeof(float)*size[0] * size[1] * size[2], cudaMemcpyHostToDevice);   // 将 temp（3） 放在GPU中
	//	cudaMemcpy(d_prj3d, projection[0][0], sizeof(float) * dsize[0]* dsize[1] * nview, cudaMemcpyHostToDevice);  // 将projection（3） 放在GPU中
	//	////////////////////////////////////////
	//	// 2D projection each view...
	//	///////////////////////////////////////
	//	
	//	for ( n = 0; n < nview; n++)  // nview:100
	//	{
	//		sin_value = h_sine[n];
	//		cos_value = h_cosin[n];
	//		//printf("sin_value = %f \n",h_sine[n]);
	//		forwardProj2d << <nblocks, NTHREAD_PER_BLOCK >> > (d_prj2d, d_temp, nx, nz, VOXSIZE_X, VOXSIZE_Z, sin_value, cos_value);
	//		cudaThreadSynchronize();
	//		cudaMemcpy(&d_prj3d[n * dsize[0] * dsize[1]], d_prj2d, sizeof(float)*dsize[0] * dsize[1], cudaMemcpyDeviceToDevice); //将每个投影d_prj2d(2)放在d_prj3d(3)中
	//	}

	//	cudaMemcpy(h_proj[0][0], d_prj3d, sizeof(float)*dsize[0] * dsize[1] * dsize[2], cudaMemcpyDeviceToHost);  // 将d_prj3d(3) 复制到 h_proj(3)中
	//	printf("projection:%d,%d,%d\n",dsize[0],dsize[1],dsize[2]);
	//	
	//	strcpy (str_outfilename,"D:/Dateset/dang/projection/Projection_");
	//	itoa(i+990,str_number,10);  // itoa()函数把整数转换成字符串，并返回指向转换后的字符串的指针
	//	strcat(str_outfilename,str_number);
	//	IMGWRITE(str_outfilename, h_proj[0][0], datasetp);
	//	
	//	printf("The simulated projection has been saved in: %s\n", str_number);
	//}





	//***********释放数据空间************************************************
	free(CT_deform[0][0]); free(CT_deform[0]); free(CT_deform);
	free(warps[0][0]); free(warps[0]); free(warps);
	free(h_angle);
	free(x3d_moving);
	free(h_proj[0][0]); free(h_proj[0]); free(h_proj);

	cudaFree(d_prj2d);
	cudaFree(d_prj3d);
	cudaFree(d_temp);
	cudaThreadExit();
	system("pause");
	return 0;


}






//********** ***************************************************************



















/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void transformvolumeGPU(float* Iin, float* Iout, float* Txyz, int* Isize_d)
{
	float* d_Iin, * d_Iout;
	int voxelNumber = Isize_d[0] * Isize_d[1] * Isize_d[2];
	checkCudaErrors(cudaMalloc((void**)&d_Iin, sizeof(float) * voxelNumber));
	checkCudaErrors(cudaMalloc((void**)&d_Iout, sizeof(float) * voxelNumber));
	checkCudaErrors(cudaMemset(d_Iout, 0, sizeof(float) * voxelNumber));
	checkCudaErrors(cudaMemcpy(d_Iin, Iin, sizeof(float) * voxelNumber, cudaMemcpyHostToDevice));
	float* d_Txyz;
	checkCudaErrors(cudaMalloc((void**)&d_Txyz, sizeof(float) * 3 * voxelNumber));
	checkCudaErrors(cudaMemcpy(d_Txyz, Txyz, sizeof(float) * 3 * voxelNumber, cudaMemcpyHostToDevice));


	double* Nthreadsd;
	int Nthreads;
	////*  if one outside pixels are set to zero. */
	double* moded;
	int mode = 0;

	/* Cubic and outside black booleans */
	int black, cubic;

	/* 3D index storage*/
	int indexI;

	/* Size of input image */
	// double *Isize_d;
	int Isize[3] = { 0,0,0 };


	/* offset */
	int ThreadOffset = 0;

	/* The thread ID number/name */
	double* ThreadID;

	/* X,Y coordinates of current pixel */
	int x, y, z;
	int xyzIndex, totalVoxel;


	Nthreads = 1;
	cubic = 0;
	black = 0;

	Isize[0] = (int)Isize_d[0];
	Isize[1] = (int)Isize_d[1];
	Isize[2] = (int)Isize_d[2];

	int* d_Isize;
	checkCudaErrors(cudaMalloc((void**)&d_Isize, sizeof(int) * 3));
	checkCudaErrors(cudaMemcpy(d_Isize, Isize, sizeof(int) * 3, cudaMemcpyHostToDevice));

	dim3 nblocks;
	int N = Isize[0] * Isize[1] * Isize[2];
	nblocks.x = NBLOCKX;
	nblocks.y = (((N - 1) / NTHREAD_PER_BLOCK)) / NBLOCKX + 1;
	interpolate_3d_double_gray_cu << < nblocks, NTHREAD_PER_BLOCK >> > (d_Iout, d_Iin, d_Txyz, d_Isize, cubic, black);

	cudaThreadSynchronize();

	cudaMemcpy(Iout, d_Iout, sizeof(float) * voxelNumber, cudaMemcpyDeviceToHost);

	cudaFree(d_Iin);
	cudaFree(d_Iout);
	cudaFree(d_Txyz);
	cudaFree(d_Isize);


}


__global__ void interpolate_3d_double_gray_cu(float* d_Iout, float* d_Iin, float* d_Txyz, int* d_Isize, int cubic, int black)
{
	const int tid = (blockIdx.y * NBLOCKX + blockIdx.x) * blockDim.x + threadIdx.x;
	int	nx = d_Isize[0];
	int ny = d_Isize[1];
	int nz = d_Isize[2];
	if (tid < nx * ny * nz)
	{
		int x = tid % nx;
		int y = ((tid - x) / nx) % ny;
		int z = tid / (nx * ny);
		// coordinate of a voxel in the phantom 

		int xyzIndex = 0;

		/* Location of translated pixel */
		double Tlocalx = 0;
		double Tlocaly = 0;
		double Tlocalz = 0;

		int totalVoxel = nx * ny * nz;

		xyzIndex = x + y * nx + z * nx * ny;

		Tlocalx = ((double)x) + d_Txyz[xyzIndex];
		Tlocaly = ((double)y) + d_Txyz[xyzIndex + totalVoxel];
		Tlocalz = ((double)z) + d_Txyz[xyzIndex + 2 * totalVoxel];

		int indexI = 0;
		indexI = z * nx * ny + y * nx + x;
		d_Iout[indexI] = interpolate_3d_double_gray_core_cu(Tlocalx, Tlocaly, Tlocalz, d_Isize, d_Iin, cubic, black);


	}
}

__device__ double interpolate_3d_double_gray_core_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* d_Isize, float* Iin, int cubic, int black)
{
	//int Isize[3] = {0,0,0};
	//Isize[0] = nx;
	//Isize[1] = ny;
	//Isize[2] = nz;

	double Ipixel;
	if (cubic) {
		if (black) { Ipixel = interpolate_3d_cubic_black_cu(Tlocalx, Tlocaly, Tlocalz, d_Isize, Iin); }
		else { Ipixel = interpolate_3d_cubic_cu(Tlocalx, Tlocaly, Tlocalz, d_Isize, Iin); }
	}
	else {
		if (black) { Ipixel = interpolate_3d_linear_black_cu(Tlocalx, Tlocaly, Tlocalz, d_Isize, Iin); }
		else { Ipixel = interpolate_3d_linear_cu(Tlocalx, Tlocaly, Tlocalz, d_Isize, Iin); }
	}
	return Ipixel;
}

__device__ double interpolate_3d_cubic_black_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* Isize, float* Iin)
{

	/* Floor of coordinate */
	double fTlocalx, fTlocaly, fTlocalz;
	/* Zero neighbor */
	int xBas0, yBas0, zBas0;
	/* The location in between the pixels 0..1 */
	double tx, ty, tz;
	/* Neighbor loccations */
	int xn[4], yn[4], zn[4];

	/* The vectors */
	double vector_tx[4], vector_ty[4], vector_tz[4];
	double vector_qx[4], vector_qy[4], vector_qz[4];
	/* Interpolated Intensity; */
	double Ipixelx = 0, Ipixelxy = 0, Ipixelxyz = 0;
	/* Loop variable */
	int i, j;

	/* Determine of the zero neighbor */
	fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly); fTlocalz = floor(Tlocalz);
	xBas0 = (int)fTlocalx; yBas0 = (int)fTlocaly; zBas0 = (int)fTlocalz;

	/* Determine the location in between the pixels 0..1 */
	tx = Tlocalx - fTlocalx; ty = Tlocaly - fTlocaly; tz = Tlocalz - fTlocalz;

	/* Determine the t vectors */
	vector_tx[0] = 0.5; vector_tx[1] = 0.5 * tx; vector_tx[2] = 0.5 * pow2_cu(tx); vector_tx[3] = 0.5 * pow3_cu(tx);
	vector_ty[0] = 0.5; vector_ty[1] = 0.5 * ty; vector_ty[2] = 0.5 * pow2_cu(ty); vector_ty[3] = 0.5 * pow3_cu(ty);
	vector_tz[0] = 0.5; vector_tz[1] = 0.5 * tz; vector_tz[2] = 0.5 * pow2_cu(tz); vector_tz[3] = 0.5 * pow3_cu(tz);

	/* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
	vector_qx[0] = -1.0 * vector_tx[1] + 2.0 * vector_tx[2] - 1.0 * vector_tx[3];
	vector_qx[1] = 2.0 * vector_tx[0] - 5.0 * vector_tx[2] + 3.0 * vector_tx[3];
	vector_qx[2] = 1.0 * vector_tx[1] + 4.0 * vector_tx[2] - 3.0 * vector_tx[3];
	vector_qx[3] = -1.0 * vector_tx[2] + 1.0 * vector_tx[3];
	vector_qy[0] = -1.0 * vector_ty[1] + 2.0 * vector_ty[2] - 1.0 * vector_ty[3];
	vector_qy[1] = 2.0 * vector_ty[0] - 5.0 * vector_ty[2] + 3.0 * vector_ty[3];
	vector_qy[2] = 1.0 * vector_ty[1] + 4.0 * vector_ty[2] - 3.0 * vector_ty[3];
	vector_qy[3] = -1.0 * vector_ty[2] + 1.0 * vector_ty[3];
	vector_qz[0] = -1.0 * vector_tz[1] + 2.0 * vector_tz[2] - 1.0 * vector_tz[3];
	vector_qz[1] = 2.0 * vector_tz[0] - 5.0 * vector_tz[2] + 3.0 * vector_tz[3];
	vector_qz[2] = 1.0 * vector_tz[1] + 4.0 * vector_tz[2] - 3.0 * vector_tz[3];
	vector_qz[3] = -1.0 * vector_tz[2] + 1.0 * vector_tz[3];

	/* Determine 1D neighbour coordinates */
	xn[0] = xBas0 - 1; xn[1] = xBas0; xn[2] = xBas0 + 1; xn[3] = xBas0 + 2;
	yn[0] = yBas0 - 1; yn[1] = yBas0; yn[2] = yBas0 + 1; yn[3] = yBas0 + 2;
	zn[0] = zBas0 - 1; zn[1] = zBas0; zn[2] = zBas0 + 1; zn[3] = zBas0 + 2;

	/* First do interpolation in the x direction followed by interpolation in the y direction */
	for (j = 0; j < 4; j++) {
		Ipixelxy = 0;
		if ((zn[j] >= 0) && (zn[j] < Isize[2])) {
			for (i = 0; i < 4; i++) {
				Ipixelx = 0;
				if ((yn[i] >= 0) && (yn[i] < Isize[1])) {
					if ((xn[0] >= 0) && (xn[0] < Isize[0])) {
						Ipixelx += vector_qx[0] * getcolor_mindex3_cu(xn[0], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
					}
					if ((xn[1] >= 0) && (xn[1] < Isize[0])) {
						Ipixelx += vector_qx[1] * getcolor_mindex3_cu(xn[1], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
					}
					if ((xn[2] >= 0) && (xn[2] < Isize[0])) {
						Ipixelx += vector_qx[2] * getcolor_mindex3_cu(xn[2], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
					}
					if ((xn[3] >= 0) && (xn[3] < Isize[0])) {
						Ipixelx += vector_qx[3] * getcolor_mindex3_cu(xn[3], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
					}
				}
				Ipixelxy += vector_qy[i] * Ipixelx;
			}
			Ipixelxyz += vector_qz[j] * Ipixelxy;
		}
	}
	return Ipixelxyz;
}

__device__ double interpolate_3d_cubic_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* Isize, float* Iin) {
	/* Floor of coordinate */
	double fTlocalx, fTlocaly, fTlocalz;
	/* Zero neighbor */
	int xBas0, yBas0, zBas0;
	/* The location in between the pixels 0..1 */
	double tx, ty, tz;
	/* Neighbor loccations */
	int xn[4], yn[4], zn[4];

	/* The vectors */
	double vector_tx[4], vector_ty[4], vector_tz[4];
	double vector_qx[4], vector_qy[4], vector_qz[4];
	/* Interpolated Intensity; */
	double Ipixelx = 0, Ipixelxy = 0, Ipixelxyz = 0;
	/* Temporary value boundary */
	int b;
	/* Loop variable */
	int i, j;

	/* Determine of the zero neighbor */
	fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly); fTlocalz = floor(Tlocalz);
	xBas0 = (int)fTlocalx; yBas0 = (int)fTlocaly; zBas0 = (int)fTlocalz;

	/* Determine the location in between the pixels 0..1 */
	tx = Tlocalx - fTlocalx; ty = Tlocaly - fTlocaly; tz = Tlocalz - fTlocalz;

	/* Determine the t vectors */
	vector_tx[0] = 0.5; vector_tx[1] = 0.5 * tx; vector_tx[2] = 0.5 * pow2_cu(tx); vector_tx[3] = 0.5 * pow3_cu(tx);
	vector_ty[0] = 0.5; vector_ty[1] = 0.5 * ty; vector_ty[2] = 0.5 * pow2_cu(ty); vector_ty[3] = 0.5 * pow3_cu(ty);
	vector_tz[0] = 0.5; vector_tz[1] = 0.5 * tz; vector_tz[2] = 0.5 * pow2_cu(tz); vector_tz[3] = 0.5 * pow3_cu(tz);

	/* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
	vector_qx[0] = -1.0 * vector_tx[1] + 2.0 * vector_tx[2] - 1.0 * vector_tx[3];
	vector_qx[1] = 2.0 * vector_tx[0] - 5.0 * vector_tx[2] + 3.0 * vector_tx[3];
	vector_qx[2] = 1.0 * vector_tx[1] + 4.0 * vector_tx[2] - 3.0 * vector_tx[3];
	vector_qx[3] = -1.0 * vector_tx[2] + 1.0 * vector_tx[3];
	vector_qy[0] = -1.0 * vector_ty[1] + 2.0 * vector_ty[2] - 1.0 * vector_ty[3];
	vector_qy[1] = 2.0 * vector_ty[0] - 5.0 * vector_ty[2] + 3.0 * vector_ty[3];
	vector_qy[2] = 1.0 * vector_ty[1] + 4.0 * vector_ty[2] - 3.0 * vector_ty[3];
	vector_qy[3] = -1.0 * vector_ty[2] + 1.0 * vector_ty[3];
	vector_qz[0] = -1.0 * vector_tz[1] + 2.0 * vector_tz[2] - 1.0 * vector_tz[3];
	vector_qz[1] = 2.0 * vector_tz[0] - 5.0 * vector_tz[2] + 3.0 * vector_tz[3];
	vector_qz[2] = 1.0 * vector_tz[1] + 4.0 * vector_tz[2] - 3.0 * vector_tz[3];
	vector_qz[3] = -1.0 * vector_tz[2] + 1.0 * vector_tz[3];

	/* Determine 1D neighbour coordinates */
	xn[0] = xBas0 - 1; xn[1] = xBas0; xn[2] = xBas0 + 1; xn[3] = xBas0 + 2;
	yn[0] = yBas0 - 1; yn[1] = yBas0; yn[2] = yBas0 + 1; yn[3] = yBas0 + 2;
	zn[0] = zBas0 - 1; zn[1] = zBas0; zn[2] = zBas0 + 1; zn[3] = zBas0 + 2;

	/* Clamp to boundary */
	if (xn[0] < 0) { xn[0] = 0; if (xn[1] < 0) { xn[1] = 0; if (xn[2] < 0) { xn[2] = 0; if (xn[3] < 0) { xn[3] = 0; } } } }
	if (yn[0] < 0) { yn[0] = 0; if (yn[1] < 0) { yn[1] = 0; if (yn[2] < 0) { yn[2] = 0; if (yn[3] < 0) { yn[3] = 0; } } } }
	if (zn[0] < 0) { zn[0] = 0; if (zn[1] < 0) { zn[1] = 0; if (zn[2] < 0) { zn[2] = 0; if (zn[3] < 0) { zn[3] = 0; } } } }
	b = Isize[0] - 1;
	if (xn[3] > b) { xn[3] = b; if (xn[2] > b) { xn[2] = b; if (xn[1] > b) { xn[1] = b; if (xn[0] > b) { xn[0] = b; } } } }
	b = Isize[1] - 1;
	if (yn[3] > b) { yn[3] = b; if (yn[2] > b) { yn[2] = b; if (yn[1] > b) { yn[1] = b; if (yn[0] > b) { yn[0] = b; } } } }
	b = Isize[2] - 1;
	if (zn[3] > b) { zn[3] = b; if (zn[2] > b) { zn[2] = b; if (zn[1] > b) { zn[1] = b; if (zn[0] > b) { zn[0] = b; } } } }


	/* First do interpolation in the x direction followed by interpolation in the y direction */
	for (j = 0; j < 4; j++) {
		Ipixelxy = 0;
		for (i = 0; i < 4; i++) {
			Ipixelx = 0;
			Ipixelx += vector_qx[0] * getcolor_mindex3_cu(xn[0], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
			Ipixelx += vector_qx[1] * getcolor_mindex3_cu(xn[1], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
			Ipixelx += vector_qx[2] * getcolor_mindex3_cu(xn[2], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
			Ipixelx += vector_qx[3] * getcolor_mindex3_cu(xn[3], yn[i], zn[j], Isize[0], Isize[1], Isize[2], Iin);
			Ipixelxy += vector_qy[i] * Ipixelx;
		}
		Ipixelxyz += vector_qz[j] * Ipixelxy;
	}
	return Ipixelxyz;
}

__device__ double interpolate_3d_linear_black_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* Isize, float* Iin) {
	double Iout;
	/*  Linear interpolation variables */
	int xBas0, xBas1, yBas0, yBas1, zBas0, zBas1;
	double perc[8];
	double xCom, yCom, zCom;
	double xComi, yComi, zComi;
	double color[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	double fTlocalx, fTlocaly, fTlocalz;

	fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly); fTlocalz = floor(Tlocalz);

	/* Determine the coordinates of the pixel(s) which will be come the current pixel */
	/* (using linear interpolation) */
	xBas0 = (int)fTlocalx; yBas0 = (int)fTlocaly; zBas0 = (int)fTlocalz;
	xBas1 = xBas0 + 1;      yBas1 = yBas0 + 1;      zBas1 = zBas0 + 1;


	color[0] = 0; color[1] = 0; color[2] = 0; color[3] = 0;
	color[4] = 0; color[5] = 0; color[6] = 0; color[7] = 0;

	if ((xBas0 >= 0) && (xBas0 < Isize[0])) {
		if ((yBas0 >= 0) && (yBas0 < Isize[1])) {
			if ((zBas0 >= 0) && (zBas0 < Isize[2])) {
				color[0] = getcolor_mindex3_cu(xBas0, yBas0, zBas0, Isize[0], Isize[1], Isize[2], Iin);
			}
			if ((zBas1 >= 0) && (zBas1 < Isize[2])) {
				color[1] = getcolor_mindex3_cu(xBas0, yBas0, zBas1, Isize[0], Isize[1], Isize[2], Iin);
			}
		}
		if ((yBas1 >= 0) && (yBas1 < Isize[1])) {
			if ((zBas0 >= 0) && (zBas0 < Isize[2])) {
				color[2] = getcolor_mindex3_cu(xBas0, yBas1, zBas0, Isize[0], Isize[1], Isize[2], Iin);
			}
			if ((zBas1 >= 0) && (zBas1 < Isize[2])) {
				color[3] = getcolor_mindex3_cu(xBas0, yBas1, zBas1, Isize[0], Isize[1], Isize[2], Iin);
			}
		}
	}
	if ((xBas1 >= 0) && (xBas1 < Isize[0])) {
		if ((yBas0 >= 0) && (yBas0 < Isize[1])) {
			if ((zBas0 >= 0) && (zBas0 < Isize[2])) {
				color[4] = getcolor_mindex3_cu(xBas1, yBas0, zBas0, Isize[0], Isize[1], Isize[2], Iin);
			}
			if ((zBas1 >= 0) && (zBas1 < Isize[2])) {
				color[5] = getcolor_mindex3_cu(xBas1, yBas0, zBas1, Isize[0], Isize[1], Isize[2], Iin);
			}
		}
		if ((yBas1 >= 0) && (yBas1 < Isize[1])) {
			if ((zBas0 >= 0) && (zBas0 < Isize[2])) {
				color[6] = getcolor_mindex3_cu(xBas1, yBas1, zBas0, Isize[0], Isize[1], Isize[2], Iin);
			}
			if ((zBas1 >= 0) && (zBas1 < Isize[2])) {
				color[7] = getcolor_mindex3_cu(xBas1, yBas1, zBas1, Isize[0], Isize[1], Isize[2], Iin);
			}
		}
	}

	/* Linear interpolation constants (percentages) */
	xCom = Tlocalx - fTlocalx;  yCom = Tlocaly - fTlocaly;   zCom = Tlocalz - fTlocalz;

	xComi = (1 - xCom); yComi = (1 - yCom); zComi = (1 - zCom);
	perc[0] = xComi * yComi; perc[1] = perc[0] * zCom; perc[0] = perc[0] * zComi;
	perc[2] = xComi * yCom;  perc[3] = perc[2] * zCom; perc[2] = perc[2] * zComi;
	perc[4] = xCom * yComi;  perc[5] = perc[4] * zCom; perc[4] = perc[4] * zComi;
	perc[6] = xCom * yCom;   perc[7] = perc[6] * zCom; perc[6] = perc[6] * zComi;

	/* Set the current pixel value */
	Iout = color[0] * perc[0] + color[1] * perc[1] + color[2] * perc[2] + color[3] * perc[3] + color[4] * perc[4] + color[5] * perc[5] + color[6] * perc[6] + color[7] * perc[7];
	return Iout;
}

__device__ double interpolate_3d_linear_cu(double Tlocalx, double Tlocaly, double Tlocalz, int* Isize, float* Iin)
{
	double Iout;
	/*  Linear interpolation variables */
	int xBas0, xBas1, yBas0, yBas1, zBas0, zBas1;
	double perc[8];
	double xCom, yCom, zCom;
	double xComi, yComi, zComi;
	double color[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	double fTlocalx, fTlocaly, fTlocalz;

	fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly); fTlocalz = floor(Tlocalz);

	/* Determine the coordinates of the pixel(s) which will be come the current pixel */
	/* (using linear interpolation) */
	xBas0 = (int)fTlocalx; yBas0 = (int)fTlocaly; zBas0 = (int)fTlocalz;
	xBas1 = xBas0 + 1;      yBas1 = yBas0 + 1;      zBas1 = zBas0 + 1;

	/* Clamp to boundary */
	if (xBas0 < 0) { xBas0 = 0; if (xBas1 < 0) { xBas1 = 0; } }
	if (yBas0 < 0) { yBas0 = 0; if (yBas1 < 0) { yBas1 = 0; } }
	if (zBas0 < 0) { zBas0 = 0; if (zBas1 < 0) { zBas1 = 0; } }
	if (xBas1 > (Isize[0] - 1)) { xBas1 = Isize[0] - 1; if (xBas0 > (Isize[0] - 1)) { xBas0 = Isize[0] - 1; } }
	if (yBas1 > (Isize[1] - 1)) { yBas1 = Isize[1] - 1; if (yBas0 > (Isize[1] - 1)) { yBas0 = Isize[1] - 1; } }
	if (zBas1 > (Isize[2] - 1)) { zBas1 = Isize[2] - 1; if (zBas0 > (Isize[2] - 1)) { zBas0 = Isize[2] - 1; } }

	/*  Get intensities */
	color[0] = getcolor_mindex3_cu(xBas0, yBas0, zBas0, Isize[0], Isize[1], Isize[2], Iin);
	color[1] = getcolor_mindex3_cu(xBas0, yBas0, zBas1, Isize[0], Isize[1], Isize[2], Iin);
	color[2] = getcolor_mindex3_cu(xBas0, yBas1, zBas0, Isize[0], Isize[1], Isize[2], Iin);
	color[3] = getcolor_mindex3_cu(xBas0, yBas1, zBas1, Isize[0], Isize[1], Isize[2], Iin);
	color[4] = getcolor_mindex3_cu(xBas1, yBas0, zBas0, Isize[0], Isize[1], Isize[2], Iin);
	color[5] = getcolor_mindex3_cu(xBas1, yBas0, zBas1, Isize[0], Isize[1], Isize[2], Iin);
	color[6] = getcolor_mindex3_cu(xBas1, yBas1, zBas0, Isize[0], Isize[1], Isize[2], Iin);
	color[7] = getcolor_mindex3_cu(xBas1, yBas1, zBas1, Isize[0], Isize[1], Isize[2], Iin);

	/* Linear interpolation constants (percentages) */
	xCom = Tlocalx - fTlocalx;  yCom = Tlocaly - fTlocaly;   zCom = Tlocalz - fTlocalz;

	xComi = (1 - xCom); yComi = (1 - yCom); zComi = (1 - zCom);
	perc[0] = xComi * yComi; perc[1] = perc[0] * zCom; perc[0] = perc[0] * zComi;
	perc[2] = xComi * yCom;  perc[3] = perc[2] * zCom; perc[2] = perc[2] * zComi;
	perc[4] = xCom * yComi;  perc[5] = perc[4] * zCom; perc[4] = perc[4] * zComi;
	perc[6] = xCom * yCom;   perc[7] = perc[6] * zCom; perc[6] = perc[6] * zComi;

	/* Set the current pixel value */
	Iout = color[0] * perc[0] + color[1] * perc[1] + color[2] * perc[2] + color[3] * perc[3] + color[4] * perc[4] + color[5] * perc[5] + color[6] * perc[6] + color[7] * perc[7];
	return Iout;
}

__device__ double pow2_cu(double val) { return val * val; }
__device__ double pow3_cu(double val) { return val * val * val; }

__device__ double getcolor_mindex3_cu(int x, int y, int z, int sizx, int sizy, int sizz, float* I)
{
	return I[z * sizx * sizy + y * sizx + x];
}

__global__ void DVF_inverse(float* d_x, int size)
{
	const int tid = (blockIdx.y * NBLOCKX + blockIdx.x) * blockDim.x + threadIdx.x;

	if (tid < size)
	{
		d_x[tid] = -d_x[tid];
	}
}

__global__ void fprj2d(float* dest, float* phantom, int NPRJ, int nx, int nz, float vx, float sin_value, float cos_value,
	int* d_vdim, float* d_length,float SID ,float SOD,int NI_X,int NI_Z)
{
	const int tid = (blockIdx.y * NBLOCKX + blockIdx.x) * blockDim.x + threadIdx.x;

	if (tid < 1 * NI_Z * NI_X)
	{
		int iimagerx = tid % NI_X;
		int iimagerz = tid / NI_X;

		float from[3], to[3];
		float xs = SOD * cos_value + nx / 2;
		float ys = SOD * sin_value + nx / 2;
		float zs = nz / 2.;
		from[0] = xs;
		from[1] = ys;
		from[2] = zs;
		// source position

		float s = (iimagerx - NI_X / 2 + 0.5) * PIXELSIZE;
		float z = (iimagerz - NI_Z / 2 + 0.5) * PIXELSIZE;
		float xi = -(SID - SOD) * cos_value - s * sin_value + nx / 2;
		float yi = -(SID - SOD) * sin_value + s * cos_value + nx / 2;
		float zi = z + nz / 2;
		to[0] = xi;
		to[1] = yi;
		to[2] = zi;
		// detector bin position


		int ssize[3], vox[3];
		ssize[0] = nx;
		ssize[1] = nx;
		ssize[2] = nz;

		int direc[3];
		int num = 0;
		dest[tid] = frt3d_kernel(ssize, phantom, from, to, vox, direc, d_vdim, d_length, nx, nz);
		//num  = frt3d_kernel(ssize, from, to, vox, direc, d_vdim, d_length); 

	}


}

__device__ int frt3d_kernel(int* ssize, float* phantom, float* from, float* to, int* vox, int* direc, int* vdim, float* length, int nx, int nz)
{
	const int D = 3;
	int num;
	float totlen, lmin, lmax, xmin;
	float dx[D];
	int ux[D];
	double k[D], l[D], lambda;  /** to trace more accurately **/
	float dum0, dum1, dum;
	int i, d;
	int i0, v[3];

	int ii;
	float sum;

	/** setup **/

	for (i = 0; i < D; i++)
	{
		dx[i] = to[i] - from[i];
		if (dx[i] > EPS_frd3d)       direc[i] = 1;
		else if (dx[i] < -EPS_frd3d) direc[i] = -1;
		else                 direc[i] = 0;
	}

	for (i = 0, totlen = 0.; i < D; i++)
		totlen += dx[i] * dx[i];

	totlen = SQRT(totlen);

	if (totlen <= 0.)
		return 0;

	//---------------------------------initialize -------------------------------------------------//

	lmin = 0.;
	lmax = totlen;
	for (i = 0; i < D; i++)
	{
		if (direc[i])
		{
			k[i] = totlen / dx[i];
			dum0 = k[i] * (-from[i]);
			dum1 = k[i] * (ssize[i] - from[i]);
			if (dum0 > dum1) SWAP(dum0, dum1, dum);
			if (lmin < dum0) lmin = dum0;
			if (lmax > dum1) lmax = dum1;
		}
		else
			if (from[i] <= 0 || from[i] >= ssize[i]) return 0.;
	}
	if (lmin >= lmax)
		return 0;

	for (i = 0; i < D; i++)
	{
		xmin = from[i] + lmin * dx[i] / totlen;
		vox[i] = (int)floorf(xmin);
		if (xmin == vox[i] && direc[i] > 0) vox[i]--;

		if (direc[i] > 0) ux[i] = (int)ceilf(xmin);

		else ux[i] = (int)floorf(xmin);

		if (direc[i]) l[i] = (ux[i] - from[i]) * k[i];
		else          l[i] = lmax + lmax;  // set as too large 
	}


	for (i = 0; i < D; i++)
	{
		if (k[i] < 0)
			k[i] = -k[i];
	}


	for (num = 0, lambda = lmin; ; )
	{
		for (d = 0, i = 1; i < D; i++)
			if (l[i] < l[d])
				d = i;
		if (l[d] >= lmax) // last voxel  
		{
			length[num] = lmax - lambda;
			vdim[++num] = d;
			break;
		}


		length[num] = l[d] - lambda;
		vdim[++num] = d;

		lambda = l[d];
		l[d] += k[d];



	}
	// return num;
	//////////////////////////////////////////////////////////////////////////

	//	 for ( i0=0; i0<num && length[i0]<1.0E-5;  d=vdim[++i0], vox[d]+=direc[d]) ;
	//for (; i0<num && length[num-1]<1.0E-5; num--);
	//	  	 
	//		 sum = 0.;
	//	 for ( i=i0, v[0]=vox[0], v[1]=vox[1], v[2]=vox[2]; i<num; d=vdim[++i], v[d]+=direc[d])
	//	 {
	//		 if(v[0]>-1&&v[1]>-1&&v[2]>-1&&v[0]<ssize[0]&&v[1]<ssize[1]&&v[2]<ssize[2])
	//		 {
	//			 //sum += phantom[v[2]][v[1]][v[0]]*length[i];
	//			 //sum += phantom[v[2]*nx*nx + v[1]*nx + v[0]]*length[i];	
	//			 sum = sum + length[i];
	//			 //sum += phantom[v[2]*ssize[0]*ssize[0] + v[1]*ssize[0] + v[0]];
	//			 
	//		 }
	//	 }


	sum = 0.;
	for (ii = 0; ii < num; ii++)
	{
		sum = sum + length[ii];
	}

	return sum;


}


__global__ void forwardProj2d(float* dest_projection, float* phantom, int nx, int nz, float vx, float vz, float sine, float cosin,float SID,float SOD,int NI_X,int NI_Z)
{
	float L1 = SOD;
	float L2 = SID - SOD;
	const int tid = (blockIdx.y * NBLOCKX + blockIdx.x) * blockDim.x + threadIdx.x;

	if (tid < NI_X * NI_Z) // for each projection
	{

		int iimagerx = tid % NI_X;
		int iimagerz = tid / NI_X;


		float xs = -L1 * cosin;
		float ys = -L1 * sine;
		float zs = 0.0F;
		// coordinate for source point 

		float xtemp = L2;
		float ytemp = -(iimagerx - NI_X / 2 + 0.5F) * PIXSIZE_X;
		float xi = xtemp * cosin - ytemp * sine;
		float yi = xtemp * sine + ytemp * cosin;
		float zi = (iimagerz - NI_Z / 2 + 0.5F) * PIXSIZE_Z;
		// coordinate for pixel on the imager



		float temp;
		int startDim;
		float alphaMin = 0.0F;
		float alphaMax = 1.0F;
		float alpha1, alphan;

		alpha1 = (-nx / 2 * vx - xs) / (xi - xs);
		alphan = (nx / 2 * vx - xs) / (xi - xs);
		temp = min(alpha1, alphan);
		alphaMin = max(alphaMin, temp);
		if (alphaMin == temp) startDim = 0;
		temp = max(alpha1, alphan);
		alphaMax = min(alphaMax, temp);

		alpha1 = (-nx / 2 * vx - ys) / (yi - ys);
		alphan = (nx / 2 * vx - ys) / (yi - ys);
		temp = min(alpha1, alphan);
		alphaMin = max(alphaMin, temp);
		if (alphaMin == temp) startDim = 1;
		temp = max(alpha1, alphan);
		alphaMax = min(alphaMax, temp);

		alpha1 = (-nz / 2 * vz - zs) / (zi - zs);
		alphan = (nz / 2 * vz - zs) / (zi - zs);
		temp = min(alpha1, alphan);
		alphaMin = max(alphaMin, temp);
		if (alphaMin == temp) startDim = 2;
		temp = max(alpha1, alphan);
		alphaMax = min(alphaMax, temp);
		//      determin the alpha range, and start dimension

		float length = sqrt((xs - xi) * (xs - xi) + (ys - yi) * (ys - yi) + (zs - zi) * (zs - zi));
		//  the total length from source to pixel


		int direction[3] = { (xi < xs) * (-2) + 1, (yi < ys) * (-2) + 1, (zi < zs) * (-2) + 1 };
		//      direction of the propagation

		float dAlpha[3] = { vx / abs(xi - xs), vx / abs(yi - ys), vz / abs(zi - zs) };
		//      determin the increment of the alphas

		dest_projection[tid] = forwardProjRay(phantom, alphaMin, alphaMax, startDim, direction, dAlpha, length, xi, yi, zi, xs, ys, zs, nx, nz, vx, vz);

		//      get forward projection along the ray line


	}


}

__device__ float forwardProjRay(float* phantom, float alphaMin, float alphaMax, int startDim, int direction[3], float dAlpha[3], float length, float xi, float yi, float zi, float xs, float ys, float zs,
	int nx, int nz, float vx, float vz)
{
	float alpha = alphaMin;
	float alphax, alphay, alphaz;
	int ix, iy, iz;
	//      variables used

	if (startDim == 0)
	{
		if (direction[0] == 1)
			ix = 0;
		//ix = nx - 1;
		else
			ix = nx - 1;
		//ix = 0;
		alphax = alpha + dAlpha[0];

		iy = (ys + alpha * (yi - ys)) / vx + nx / 2;
		alphay = ((iy + (direction[1] > 0) - nx / 2) * vx - ys) / (yi - ys);

		iz = (zs + alpha * (zi - zs)) / vz + nz / 2;
		alphaz = ((iz + (direction[2] > 0) - nz / 2) * vz - zs) / (zi - zs);
	}
	else if (startDim == 1)
	{
		if (direction[1] == 1)
			iy = 0;
		//iy = nx -1;
		else
			iy = nx - 1;
		//iy = 0;
		alphay = alpha + dAlpha[1];

		ix = (xs + alpha * (xi - xs)) / vx + nx / 2;
		alphax = ((ix + (direction[0] > 0) - nx / 2) * vx - xs) / (xi - xs);

		iz = (zs + alpha * (zi - zs)) / vz + nz / 2;
		alphaz = ((iz + (direction[2] > 0) - nz / 2) * vz - zs) / (zi - zs);
	}
	else if (startDim == 2)
	{
		if (direction[2] == 1)
			iz = 0;
		//iz = nz - 1;
		else
			iz = nz - 1;
		//  iz = 0;
		alphaz = alpha + dAlpha[2];

		ix = (xs + alpha * (xi - xs)) / vx + nx / 2;
		alphax = ((ix + (direction[0] > 0) - nx / 2) * vx - xs) / (xi - xs);

		iy = (ys + alpha * (yi - ys)) / vx + nx / 2;
		alphay = ((iy + (direction[1] > 0) - nx / 2) * vx - ys) / (yi - ys);
	}
	//      initialize the ray tracing

	float result = 0.0F;

	while (alpha < alphaMax - TOL)
		//      tracing the line while accumulate the projection
	{
		if (alphax <= alphay && alphax <= alphaz)
		{
			result += length * (alphax - alpha) * phantom[iz * nx * nx + iy * nx + ix];
			//result += length * (alphax-alpha);
			alpha = alphax;
			alphax += dAlpha[0];
			ix += direction[0];

		}
		else if (alphay <= alphaz)
		{
			result += length * (alphay - alpha) * phantom[iz * nx * nx + iy * nx + ix];
			//result += length * (alphay-alpha);
			alpha = alphay;
			alphay += dAlpha[1];
			iy += direction[1];
		}
		else
		{
			result += length * (alphaz - alpha) * phantom[iz * nx * nx + iy * nx + ix];
			//result += length * (alphaz-alpha);
			alpha = alphaz;
			alphaz += dAlpha[2];
			iz += direction[2];

		}

	}

	return result;
}