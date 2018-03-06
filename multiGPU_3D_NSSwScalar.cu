
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// includes, project
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>

// include parameters for DNS
#include "DNS_PARAMETERS.h"

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
int idxClip(int idx, int idxMax){
	return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int stack, int width, int height, int depth){
	return idxClip(stack, depth) + idxClip(row, height)*depth + idxClip(col, width)*depth*height;
	// Note: using column-major indexing format
}

void writeDouble(double v, FILE *f)  {
	fwrite((void*)(&v), sizeof(v), 1, f);

	return;
}

void displayDeviceProps(int numGPUs){
	int i, driverVersion = 0, runtimeVersion = 0;

	for( i = 0; i<numGPUs; ++i)
	{
		cudaSetDevice(i);

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("  Device name: %s\n", deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
	
		char msg[256];
		SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes \n",
				(float)deviceProp.totalGlobalMem/1048576.0f);
		printf("%s", msg);

		printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			   deviceProp.multiProcessorCount,
			   _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			   _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

		printf("\n");
	}

	return;
}

void writeData_mgpu( int nGPUs, int *NX_per_GPU, int *start_x, const int iter, const char var, double **in ) 
{
	int i, j, k, n, idx;
	char title[0x100];
	// snprintf(title, sizeof(title), SaveLocation, NX, Re, var, iter);
	snprintf(title, sizeof(title), SaveLocation, var, iter);
	printf("Saving data to %s \n", title);
	FILE *out = fopen(title, "wb");
	writeDouble(sizeof(double) * NX*NY*NZ, out);
	// writelu(sizeof(double) * NX*NY*NZ, out);
	for (n = 0; n < nGPUs; ++n){
		for (i = 0; i < NX_per_GPU[n]; ++i){
			for (j = 0; j < NY; ++j){
				for (k = 0; k < NZ; ++k){
					idx = k + 2*NZ2*j + 2*NZ2*NY*i;		// Using padded index for in-place FFT
					writeDouble(in[n][idx], out);
				}
			}
		}			
	}


	fclose(out);

	return;
}

int readDataSize(FILE *f){
	int bin;

	int flag = fread((void*)(&bin), sizeof(float), 1, f);

	if(flag == 1)
		return bin;
	else{
		return 0;
	}
}

double readDouble(FILE *f){
	double v;

	int flag = fread((void*)(&v), sizeof(double), 1, f);

	if(flag == 1)
		return v;
	else{
		return 0;
	}
}

void loadData(int nGPUs, int *start_x, int *NX_per_GPU, const char *name, double **var)
{ // Function to read in velocity data into multiple GPUs

	int i, j, k, n, idx, N;
	char title[0x100];
	snprintf(title, sizeof(title), DataLocation, name);
	printf("Reading data from %s \n", title);
	FILE *file = fopen(title, "rb");
	N = readDouble(file)/sizeof(double);
	printf("The size of N is %d\n",N);
        for (n = 0; n < nGPUs; ++n){
        	printf("Reading data for GPU %d\n",n);
                for (i = 0; i < NX_per_GPU[n]; ++i){
                        for (j = 0; j < NY; ++j){
                                for (k = 0; k < NZ; ++k){
                                        idx = k + 2*NZ2*j + 2*NZ2*NY*i;	
                                        var[n][idx] = readDouble(file);
                                }
                        }
                }
        }

	fclose(file);

	return;
}

void splitData(int numGPUs, int size, int *size_per_GPU, int *start_idx) {
	int i, n;
	if(size % numGPUs == 0){
		for (i=0;i<numGPUs;++i){
			size_per_GPU[i] = size/numGPUs;
			start_idx[i] = i*size_per_GPU[i];              
		}
	}
	else {
		printf("Warning: number of GPUs is not an even multiple of the data size\n");
		n = size/numGPUs;
		for(i=0; i<(numGPUs-1); ++i){
			size_per_GPU[i] = n;
			start_idx[i] = i*size_per_GPU[i];
		}
		size_per_GPU[numGPUs-1] = n + size % numGPUs;
		start_idx[numGPUs-1] = (numGPUs-1)*size_per_GPU[numGPUs-2];
	}
}

__global__ 
void initializeVelocityKernel_mgpu(int start_x, cufftDoubleReal *f1, cufftDoubleReal *f2, cufftDoubleReal *f3)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( (i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten( i, j, k, NX, NY, 2*NZ2);	// Index local to the GPU

	// Create physical vectors in temporary memory
	double x = (i + start_x) * (double)LX / NX;
	double y = j * (double)LY / NY;
	double z = k * (double)LZ / NZ;

	// Initialize starting array
	f1[idx] = sin(x)*cos(y)*cos(z);
	f2[idx] = -cos(x)*sin(y)*cos(z);
	f3[idx] = 0.0;

	return;
}

__global__ 
void initializeScalarKernel_mgpu(int start_x, cufftDoubleReal *Z)
{	// Creates initial conditions in the physical domain
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);	// Index local to each GPU

	// Create physical vectors in temporary memory
	double x = (i+start_x) * (double)LX / NX;

	// Initialize scalar field
	if ( (i+start_x) < NX/2 ){
		Z[idx] = 0.5 * (1 + tanh( (x - PI/2) * LX) );
	}
	else {
		Z[idx] = 0.5 * (1 - tanh( (x - 3*PI/2) * LX) );
	}

	return;
}

void initializeVelocity(int nGPUs, int *start_x, int *NX_per_GPU, cufftDoubleReal **u, cufftDoubleReal **v, cufftDoubleReal **w)
{
	int n;
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX_per_GPU[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		initializeVelocityKernel_mgpu<<<gridSize, blockSize>>>(start_x[n], u[n], v[n], w[n]);
		printf("Data initialized on GPU #%d...\n",n);
	}

	return;

}

void initializeScalar(int nGPUs, int *start_x, int *NX_per_GPU, cufftDoubleReal **z)
{
	int n;
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX_per_GPU[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		initializeScalarKernel_mgpu<<<gridSize, blockSize>>>(start_x[n], z[n]);
		printf("Data initialized on GPU #%d...\n",n);
	}

	return;

}

void importVelocity(int nGPUs, int *start_x, int *NX_per_GPU, double **h_u, double **h_v, double **h_w, cufftDoubleReal **u, cufftDoubleReal **v, cufftDoubleReal **w)
{	// Import data from file
	int n;

	loadData(nGPUs, start_x, NX_per_GPU, "u", h_u);
	loadData(nGPUs, start_x, NX_per_GPU, "v", h_v);
	loadData(nGPUs, start_x, NX_per_GPU, "w", h_w);

	// Copy data from host to device
	printf("Copy results to GPU memory...\n");
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
		checkCudaErrors( cudaMemcpyAsync(u[n], h_u[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(v[n], h_v[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(w[n], h_w[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
	}

}

void importScalar(int nGPUs, int *start_x, int *NX_per_GPU, double **h_z, cufftDoubleReal **z)
{	// Import data from file
	int n;

	loadData(nGPUs, start_x, NX_per_GPU, "z", h_z);

	// Copy data from host to device
	printf("Copy results to GPU memory...\n");
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
		checkCudaErrors( cudaMemcpyAsync(z[n], h_z[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
	}

}

__global__
void waveNumber_kernel(double *waveNum)
{   // Creates the wavenumber vectors used in Fourier space
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= NX) return;

	if (i < NX/2)
		waveNum[i] = (double)i;
	else
		waveNum[i] = (double)i - NX;

	return;
}

void initializeWaveNumbers(int nGPUs, double **waveNum)
{    // Initialize wavenumbers in Fourier space

	int n;
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		waveNumber_kernel<<<divUp(NX,TX), TX>>>(waveNum[n]);
	}

	printf("Wave domain setup complete..\n");

	return;
}

__global__
void deAliasKernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *fhat){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	double k_sq = waveNum[i]*waveNum[i] + waveNum[(j+start_y)]*waveNum[(j+start_y)] + waveNum[k]*waveNum[k];

	if( k_sq > (k_max*k_max) )
	{
		fhat[idx].x = 0.0;
		fhat[idx].y = 0.0;
	}

	return;
}

void deAlias(int nGPUs, int *NY_per_GPU, int *start_y, double **k, cufftDoubleComplex **f1hat, cufftDoubleComplex **f2hat, cufftDoubleComplex **f3hat, cufftDoubleComplex **f4hat)
{	// Truncate data for de-aliasing

	int n;
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(NY_per_GPU[n], TY), divUp(NZ2, TZ));

		// Call the kernel
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], k[n], f1hat[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], k[n], f2hat[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], k[n], f3hat[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], k[n], f4hat[n]);
	}
	
	return;
}

void plan2dFFT(int nGPUs, int *NX_per_GPU, size_t *worksize_f, size_t *worksize_i, cufftDoubleComplex **workspace, cufftHandle *plan, cufftHandle *invplan){
// This function plans a 2-dimensional FFT to operate on the X and Y directions (assumes X-direction is contiguous in memory)
    int result;

    int n;
    for(n = 0; n<nGPUs; ++n){
        cudaSetDevice(n);

        //Create plan for 2-D cuFFT, set cuFFT parameters
        int rank = 2;
        int size[] = {NY,NZ};           
        int inembed[] = {NY,2*NZ2};         // inembed measures distance between dimensions of data
        int onembed[] = {NY,NZ2};     // Uses half the domain for a R2C transform
        int istride = 1;                        // istride is distance between consecutive elements
        int ostride = 1;
        int idist = NY*2*NZ2;                      // idist is the total length of one signal
        int odist = NY*NZ2;
        int batch = NX_per_GPU[n];                        // # of 2D FFTs to perform

        // Create empty plan handles
        cufftCreate(&plan[n]);
        cufftCreate(&invplan[n]);

        // Disable auto allocation of workspace memory for cuFFT plans
        result = cufftSetAutoAllocation(plan[n], 0);
        if ( result != CUFFT_SUCCESS){
                printf("CUFFT error: cufftSetAutoAllocation failed on line %d, Error code %d\n", __LINE__, result);
        return; }
        result = cufftSetAutoAllocation(invplan[n], 0);
        if ( result != CUFFT_SUCCESS){
                printf("CUFFT error: cufftSetAutoAllocation failed on line %d, Error code %d\n", __LINE__, result);
        return; }

        // Plan Forward 2DFFT
        result = cufftMakePlanMany(plan[n], rank, size, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batch, &worksize_f[n]);
        if ( result != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: cufftPlanforward 2D failed");
            printf(", Error code %d\n", result);
        return; 
        }

        // Plan inverse 2DFFT
        result = cufftMakePlanMany(invplan[n], rank, size, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batch, &worksize_i[n]);
        if ( result != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: cufftPlanforward 2D failed");
            printf(", Error code %d\n", result);
        return; 
        }

        printf("The workspace size required for the forward transform is %lu.\n", worksize_f[n]);
        printf("The workspace size required for the inverse transform is %lu.\n", worksize_i[n]);

        // Assuming that both workspaces are the same size (seems to be generally true), then the two workspaces can share an allocation
        // Allocate workspace memory
        checkCudaErrors( cudaMalloc(&workspace[n], worksize_f[n]) );

        // Set cuFFT to use allocated workspace memory
        result = cufftSetWorkArea(plan[n], workspace[n]);
        if ( result != CUFFT_SUCCESS){
                printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, result);
        return; }
        result = cufftSetWorkArea(invplan[n], workspace[n]);
        if ( result != CUFFT_SUCCESS){
                printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, result);
        return; }    

        }

    return;
}

void plan1dFFT(int nGPUs, cufftHandle *plan){
// This function plans a 1-dimensional FFT to operate on the Z direction (assuming Z-direction is contiguous in memory)
    int result;

    int n;
    for(n = 0; n<nGPUs; ++n){
        cudaSetDevice(n);
        //Create plan for cuFFT, set cuFFT parameters
        int rank = 1;               // Dimensionality of the FFT - constant at rank 1
        int size[] = {NX};          // size of each rank
        int inembed[] = {0};            // inembed measures distance between dimensions of data
        int onembed[] = {0};       // For complex to complex transform, input and output data have same dimensions
        int istride = NZ2;                        // istride is distance between consecutive elements
        int ostride = NZ2;
        int idist = 1;                     // idist is the total length of one signal
        int odist = 1;
        int batch = NZ2;                      // # of 1D FFTs to perform (assuming data has been transformed previously in the Z-Y directions)

        // Plan Forward 1DFFT
        result = cufftPlanMany(&plan[n], rank, size, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);
        if ( result != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: cufftPlanforward failed");
        return; 
        }
    }
    
    return;
}

__global__
void scaleKernel_mgpu(int start_x, cufftDoubleReal *f)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( (i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten( i, j, k, NX, NY, 2*NZ2);

	f[idx] = f[idx] / ( (double)NX*NY*NZ );

	return;
}

__global__ 
void organizeData(cufftDoubleComplex *in, cufftDoubleComplex *out, int N, int j)
{// Function to grab non-contiguous chunks of data and make them contiguous

	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k >= NZ2) return;

	for(int i=0; i<N; ++i){

		// printf("For thread %d, indexing begins at local index of %d, which maps to temp at location %d\n", k, (k+ NZ*j), k);
		out[k + i*NZ2] = in[k + NZ2*j + i*NY*NZ2];

	}

	return;

}

void transpose_xy_mgpu(cufftDoubleComplex **src, cufftDoubleComplex **dst, cufftDoubleComplex **temp, int nGPUs)
{   // Transpose x and y directions (for a z-contiguous 1d array distributed across multiple GPUs)
	// This function loops through GPUs (instead of looping through all x,y) to do the transpose. Requires extra conversion to calculate the local index at the source location.
	// printf("Taking Transpose...\n");

	int n, j, local_idx_dst, dstNum;

   
	for(j=0; j<NY; ++j){
		for(n=0; n<nGPUs; ++n){
			cudaSetDevice(n); 

			dstNum = j*nGPUs/NY;

			// Open kernel that grabs all data 
			organizeData<<<divUp(NZ2,TX), TX>>>(src[n], temp[n], NX/nGPUs, j);

			local_idx_dst = n*NX/nGPUs*NZ2 + (j - dstNum*NY/nGPUs)*NZ2*NX;

			checkCudaErrors( cudaMemcpyAsync(&dst[dstNum][local_idx_dst], temp[n], sizeof(cufftDoubleComplex)*NZ2*NX/nGPUs, cudaMemcpyDeviceToDevice) );
		}
	}

	return;
}

void Execute1DFFT_Forward(cufftHandle plan, int NY_per_GPU, cufftDoubleComplex *f, cufftDoubleComplex *fhat)
{

	cufftResult result;
	// Loop through each slab in the Y-direction
	// Perform forward FFT
	for(int i=0; i<NY_per_GPU; ++i){
		result = cufftExecZ2Z(plan, &f[i*NZ2*NX], &fhat[i*NZ2*NX], CUFFT_FORWARD);
		if (  result != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecZ2Z failed, error code %d\n",(int)result);
		return; 
		}       
	}

	return;
}

void Execute1DFFT_Inverse(cufftHandle plan, int NY_per_GPU, cufftDoubleComplex *fhat, cufftDoubleComplex *f)
{
	cufftResult result;

	// Loop through each slab in the Y-direction
	// Perform forward FFT
	for(int i=0; i<NY_per_GPU; ++i){
		result = cufftExecZ2Z(plan, &fhat[i*NZ2*NX], &f[i*NZ2*NX], CUFFT_INVERSE);
		if (  result != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecZ2Z failed, error code %d\n",(int)result);
		return; 
		}       
	}

	return;
}

void forwardTransform(cufftHandle *p_1d, cufftHandle *p_2d, int nGPUs, int *NX_per_GPU, int *start_x, int *NY_per_GPU, int *start_y, cufftDoubleComplex **f_t, cufftDoubleComplex **temp, cufftDoubleReal **f )
{ // Transform from physical to wave domain

	int RESULT, n;

	// Take FFT in Z and Y directions
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		RESULT = cufftExecD2Z(p_2d[n], f[n], (cufftDoubleComplex *)f[n]);
		if ( RESULT != CUFFT_SUCCESS){
			printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, RESULT);
		return; }
		// printf("Taking 2D forward FFT on GPU #%2d\n",n);
	}

	// Transpose X and Y dimensions
	transpose_xy_mgpu((cufftDoubleComplex **)f, f_t, temp, nGPUs);

	// Take FFT in X direction (which has been transposed to what used to be the Y dimension)
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		Execute1DFFT_Forward(p_1d[n], NY_per_GPU[n], f_t[n], (cufftDoubleComplex *)f[n]);
		// printf("Taking 1D forward FFT on GPU #%2d\n",n);
	}

	// Results remain in transposed coordinates

	// printf("Forward Transform Completed...\n");

	return;
}

void inverseTransform(cufftHandle *invp_1d, cufftHandle *invp_2d,  int nGPUs, int *NX_per_GPU, int *start_x, int *NY_per_GPU, int *start_y, cufftDoubleComplex **f_t, cufftDoubleComplex **temp, cufftDoubleComplex **f)
{ // Transform variables from wavespace to the physical domain 
	int RESULT, n;

	// Data starts in transposed coordinates, x,y flipped

	// Take FFT in X direction (which has been transposed to what used to be the Y dimension)
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);
		Execute1DFFT_Inverse(invp_1d[n], NY_per_GPU[n], f[n], f_t[n]);
		// printf("Taking 1D inverse FFT on GPU #%2d\n",n);
	}

	// Transpose X and Y directions
	transpose_xy_mgpu(f_t, f, temp, nGPUs);

	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);
		// Take inverse FFT in Z and Y direction
		RESULT = cufftExecZ2D(invp_2d[n], f[n], (cufftDoubleReal *)f[n]);
		if ( RESULT != CUFFT_SUCCESS){
			printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, RESULT);
		return; }
		// printf("Taking 2D inverse FFT on GPU #%2d\n",n);
	}

	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX_per_GPU[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		scaleKernel_mgpu<<<gridSize, blockSize>>>(start_x[n], (cufftDoubleReal *)f[n]);
	}

	// printf("Scaled Inverse Transform Completed...\n");

	return;
}

__global__
void calcOmega1Kernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *u2hat, cufftDoubleComplex *u3hat, cufftDoubleComplex *omega1){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || ((j + start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega1[idx].x = -waveNum[(j + start_y)]*u3hat[idx].y + waveNum[k]*u2hat[idx].y;
	omega1[idx].y = waveNum[(j + start_y)]*u3hat[idx].x - waveNum[k]*u2hat[idx].x;

	return;
}

__global__
void calcOmega2Kernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *u1hat, cufftDoubleComplex *u3hat, cufftDoubleComplex *omega2){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || ((j + start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega2[idx].x = waveNum[i]*u3hat[idx].y - waveNum[k]*u1hat[idx].y;
	omega2[idx].y = -waveNum[i]*u3hat[idx].x + waveNum[k]*u1hat[idx].x;

	return;
}

__global__
void calcOmega3Kernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *u1hat, cufftDoubleComplex *u2hat, cufftDoubleComplex *omega3){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || ((j + start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega3[idx].x = -waveNum[i]*u2hat[idx].y + waveNum[(j + start_y)]*u1hat[idx].y;
	omega3[idx].y = waveNum[i]*u2hat[idx].x - waveNum[(j + start_y)]*u1hat[idx].x;

	return;
}

void calcVorticity(int nGPUs, int *NY_per_GPU, int *start_y, double **waveNum, cufftDoubleComplex **u1hat, cufftDoubleComplex **u2hat, cufftDoubleComplex **u3hat, cufftDoubleComplex **omega1, cufftDoubleComplex **omega2, cufftDoubleComplex **omega3){
	// Function to calculate the vorticity in Fourier Space and transform to physical space
	
	int n;
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(NY_per_GPU[n], TY), divUp(NZ2, TZ));

		// Call kernels to calculate vorticity
		calcOmega1Kernel_mgpu<<<gridSize, blockSize>>>(start_y[n], waveNum[n], u2hat[n], u3hat[n], omega1[n]);
		calcOmega2Kernel_mgpu<<<gridSize, blockSize>>>(start_y[n], waveNum[n], u1hat[n], u3hat[n], omega2[n]);
		calcOmega3Kernel_mgpu<<<gridSize, blockSize>>>(start_y[n], waveNum[n], u1hat[n], u2hat[n], omega3[n]);
		// Kernel calls include scaling for post-FFT
	}

	// printf("Vorticity calculated in fourier space...\n");

	return;
}

__global__
void CrossProductKernel_mgpu(int start_x, cufftDoubleReal *u1, cufftDoubleReal *u2, cufftDoubleReal *u3, cufftDoubleReal *omega1, cufftDoubleReal *omega2, cufftDoubleReal *omega3){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

	// Load values into register memory (would be overwritten if not loaded into memory)
	double w1 = omega1[idx];
	double w2 = omega2[idx];
	double w3 = omega3[idx];

	__syncthreads();

	// Direction 1
	omega1[idx] = w2*u3[idx] - w3*u2[idx];
	// Direction 2
	omega2[idx] = -w1*u3[idx] + w3*u1[idx];
	// Direction 3
	omega3[idx] = w1*u2[idx] - w2*u1[idx];

	return;
}

void formCrossProduct(int nGPUs, int *NX_per_GPU, int *start_x, cufftDoubleReal **u, cufftDoubleReal **v, cufftDoubleReal **w, cufftDoubleReal **omega1, cufftDoubleReal **omega2, cufftDoubleReal **omega3){
// Function to evaluate omega x u in physical space and then transform the result to Fourier Space

	int n;
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX_per_GPU[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		// Call kernel to calculate vorticity
		CrossProductKernel_mgpu<<<gridSize, blockSize>>>(start_x[n], u[n], v[n], w[n], omega1[n], omega2[n], omega3[n]);

		cudaDeviceSynchronize();
	}

	// printf("Cross Product calculated!\n");

	return;
}

__global__
void multIkKernel_mgpu(const int dir, int start_y, double *waveNum, cufftDoubleComplex *f, cufftDoubleComplex *fIk)
{   // Multiples an input array by ik 
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	if(dir == 1){
		fIk[idx].x = -waveNum[i]*f[idx].y;     // Scaling results for when the inverse FFT is taken
		fIk[idx].y = waveNum[i]*f[idx].x;
	}

	if(dir == 2){
		fIk[idx].x = -waveNum[j+start_y]*f[idx].y;     // Scaling results for when the inverse FFT is taken
		fIk[idx].y = waveNum[j+start_y]*f[idx].x;
	}

	if(dir == 3){
		fIk[idx].x = -waveNum[k]*f[idx].y;     // Scaling results for when the inverse FFT is taken
		fIk[idx].y = waveNum[k]*f[idx].x;
	}

	return;
}

void takeDerivative(int dir, int nGPUs, int *NY_per_GPU, int *start_y, double **waveNum, cufftDoubleComplex **f, cufftDoubleComplex **fIk)
{
	// Loop through GPUs and multiply by iK
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(NY_per_GPU[n], TY), divUp(NZ2, TZ));

		// Take derivative (dir = 1 => x-direction, 2 => y-direction, 3 => z-direction)
		multIkKernel_mgpu<<<gridSize, blockSize>>>(dir, start_y[n], waveNum[n], f[n], fIk[n]);
	}

	return;  
}

__global__
void multAndAddKernel_mgpu(int start_x, cufftDoubleReal *f1, cufftDoubleReal *f2, cufftDoubleReal *f3)
{	// Function to compute the non-linear terms on the RHS

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

	f3[idx] = f3[idx] + f1[idx] * f2[idx];
		
	return;
}

void multAndAdd(int nGPUs, int *NX_per_GPU, int *start_x, cufftDoubleReal **f1, cufftDoubleReal **f2, cufftDoubleReal **f3)
{
	// Loop through GPUs and perform operation: f1*f2 + f3 = f3
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX_per_GPU[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		// Take derivative (dir = 1 => x-direction, 2 => y-direction, 3 => z-direction)
		multAndAddKernel_mgpu<<<gridSize, blockSize>>>(start_x[n], f1[n], f2[n], f3[n]);
	}

	return;  
}

void formScalarAdvection(cufftHandle *plan1d, cufftHandle *plan2d, cufftHandle *invplan2d, int nGPUs, int *NX_per_GPU, int *start_x, int *NY_per_GPU, int *start_y, cufftDoubleComplex **temp, cufftDoubleComplex **temp_reorder, cufftDoubleComplex **temp_advective, double **k, cufftDoubleReal **u, cufftDoubleReal **v, cufftDoubleReal ** w, cufftDoubleComplex **zhat, cufftDoubleComplex **rhs_z)
{	// Compute the advection term in the scalar equation

	// Zero out right hand side term before beginning calculation
	int n;
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);
		checkCudaErrors( cudaMemset(rhs_z[n], 0, sizeof(cufftDoubleComplex)*NZ2*NX*NY_per_GPU[n]) );
	}

	//===============================================================
	// ( u \dot grad ) z = u * dZ/dx + v * dZ/dy + w * dZ/dz
	//===============================================================

	// Calculate u*dZdx and add it to RHS
	// Find du/dz in Fourier space
	takeDerivative(1, nGPUs, NY_per_GPU, start_y, k, zhat, temp_advective);
	// Transform du/dz to physical space
	inverseTransform(plan1d, invplan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, temp_advective);
	// Form term and add to RHS
	multAndAdd(nGPUs, NX_per_GPU, start_x, u, (cufftDoubleReal **)temp_advective, (cufftDoubleReal **)rhs_z);


	// Calculate v*dZdy and add it to RHS
	// Find du/dz in Fourier space
	takeDerivative(2, nGPUs, NY_per_GPU, start_y, k, zhat, temp_advective);
	// Transform du/dz to physical space
	inverseTransform(plan1d, invplan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, temp_advective);
	// Form term and add to RHS
	multAndAdd(nGPUs, NX_per_GPU, start_x, v, (cufftDoubleReal **)temp_advective, (cufftDoubleReal **)rhs_z);


	// Calculate w*dZdz and add it to RHS
	// Find du/dz in Fourier space
	takeDerivative(3, nGPUs, NY_per_GPU, start_y, k, zhat, temp_advective);
	// Transform du/dz to physical space
	inverseTransform(plan1d, invplan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, temp_advective);
	// Form term and add to RHS
	multAndAdd(nGPUs, NX_per_GPU, start_x, w, (cufftDoubleReal **)temp_advective, (cufftDoubleReal **)rhs_z);

	// rhs_z now holds the advective terms of the scalar equation (in physical domain). 
	// printf("Scalar advection terms formed...\n");

	return;
}

__global__
void computeRHSKernel_mgpu(int start_y, double *k1, cufftDoubleComplex *rhs_u1, cufftDoubleComplex *rhs_u2, cufftDoubleComplex *rhs_u3, cufftDoubleComplex *rhs_Z)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// if(i == 0 && j == 0 && k ==0){printf("Calling computeRHS kernel\n");}

	// Move RHS into register memory (otherwise the values would be overwritten)
	double temp1_r = rhs_u1[idx].x;
	double temp1_c = rhs_u1[idx].y;

	double temp2_r = rhs_u2[idx].x;
	double temp2_c = rhs_u2[idx].y;

	double temp3_r = rhs_u3[idx].x;
	double temp3_c = rhs_u3[idx].y;

	// Calculate k^2 for each index
	double k_sq = k1[i]*k1[i] + k1[(j+start_y)]*k1[(j+start_y)] + k1[k]*k1[k];

	// Form RHS
	if( i == 0 && (j+start_y) == 0 && k == 0){
		rhs_u1[idx].x = 0.0;
		rhs_u1[idx].y = 0.0;

		rhs_u2[idx].x = 0.0;
		rhs_u2[idx].y = 0.0;

		rhs_u3[idx].x = 0.0;
		rhs_u3[idx].y = 0.0;

		rhs_Z[idx].x = 0.0;
		rhs_Z[idx].y = 0.0;
	}
	else {
		rhs_u1[idx].x = (k1[i]*k1[i] / k_sq - 1.0)*temp1_r + (k1[i]*k1[(j+start_y)] / k_sq)*temp2_r + (k1[i]*k1[k] / k_sq)*temp3_r;
		rhs_u1[idx].y = (k1[i]*k1[i] / k_sq - 1.0)*temp1_c + (k1[i]*k1[(j+start_y)] / k_sq)*temp2_c + (k1[i]*k1[k] / k_sq)*temp3_c;

		rhs_u2[idx].x = (k1[(j+start_y)]*k1[i] / k_sq)*temp1_r + (k1[(j+start_y)]*k1[(j+start_y)] / k_sq - 1.0)*temp2_r + (k1[(j+start_y)]*k1[k] / k_sq)*temp3_r;
		rhs_u2[idx].y = (k1[(j+start_y)]*k1[i] / k_sq)*temp1_c + (k1[(j+start_y)]*k1[(j+start_y)] / k_sq - 1.0)*temp2_c + (k1[(j+start_y)]*k1[k] / k_sq)*temp3_c;

		rhs_u3[idx].x = (k1[k]*k1[i] / k_sq)*temp1_r + (k1[k]*k1[(j+start_y)] / k_sq)*temp2_r + (k1[k]*k1[k] / k_sq - 1.0)*temp3_r;
		rhs_u3[idx].y = (k1[k]*k1[i] / k_sq)*temp1_c + (k1[k]*k1[(j+start_y)] / k_sq)*temp2_c + (k1[k]*k1[k] / k_sq - 1.0)*temp3_c;

		rhs_Z[idx].x = -rhs_Z[idx].x;
		rhs_Z[idx].y = -rhs_Z[idx].y;
	}

	return;
}

void makeRHS(int nGPUs, int *NY_per_GPU, int *start_y, double **waveNum, cufftDoubleComplex **rhs_u1, cufftDoubleComplex **rhs_u2, cufftDoubleComplex **rhs_u3, cufftDoubleComplex **rhs_Z)
{	// Function to create the rhs of the N-S equations in Fourier Space

	int n;
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(NY_per_GPU[n], TY), divUp(NZ2, TZ));

		// Call the kernel
		computeRHSKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], waveNum[n], rhs_u1[n], rhs_u2[n], rhs_u3[n], rhs_Z[n]);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) 
	    printf("Error: %s\n", cudaGetErrorString(err));	
	}

	// printf("Right hand side of equations formed!\n");

	return;
}


__global__
void eulerKernel_mgpu(double num, int start_y, double *waveNum, cufftDoubleComplex *fhat,  cufftDoubleComplex *rhs_f)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Calculate k^2 for each index
	double k_sq = waveNum[i]*waveNum[i] + waveNum[(j+start_y)]*waveNum[(j+start_y)] + waveNum[k]*waveNum[k];

	// Timestep in X-direction
	fhat[idx].x = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].x + dt * rhs_f[idx].x ) / (1.0 + dt/2.0*k_sq/num);
	fhat[idx].y = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].y + dt * rhs_f[idx].y ) / (1.0 + dt/2.0*k_sq/num);

	return;
}

__global__
void adamsBashforthKernel_mgpu(double num, int start_y, double *waveNum, cufftDoubleComplex *fhat, cufftDoubleComplex *rhs_f, cufftDoubleComplex *rhs_f_old)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Calculate k^2 for each index
	double k_sq = waveNum[i]*waveNum[i] + waveNum[(j+start_y)]*waveNum[(j+start_y)] + waveNum[k]*waveNum[k];

	// Timestep in X-direction
	fhat[idx].x = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].x + dt * (1.5*rhs_f[idx].x - 0.5*rhs_f_old[idx].x) ) / (1.0 + dt/2.0*k_sq/num);
	fhat[idx].y = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].y + dt * (1.5*rhs_f[idx].y - 0.5*rhs_f_old[idx].y) ) / (1.0 + dt/2.0*k_sq/num);

	return;
}

void timestep(const int flag, int nGPUs, int *NY_per_GPU, int *start_y, double **k, cufftDoubleComplex **uhat, cufftDoubleComplex **rhs_u, cufftDoubleComplex **rhs_u_old, cufftDoubleComplex **vhat, cufftDoubleComplex **rhs_v, cufftDoubleComplex **rhs_v_old, cufftDoubleComplex **what, cufftDoubleComplex **rhs_w, cufftDoubleComplex **rhs_w_old, cufftDoubleComplex **zhat, cufftDoubleComplex **rhs_z, cufftDoubleComplex **rhs_z_old)
{
	int n;
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(NY_per_GPU[n], TY), divUp(NZ2, TZ));

		if(flag){
			// printf("Using Euler Method\n");
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re, start_y[n], k[n], uhat[n], rhs_u[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re, start_y[n], k[n], vhat[n], rhs_v[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re, start_y[n], k[n], what[n], rhs_w[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re*Sc, start_y[n], k[n], zhat[n], rhs_z[n]);
		}
		else {
			// printf("Using A-B Method\n");
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re, start_y[n], k[n], uhat[n], rhs_u[n], rhs_u_old[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re, start_y[n], k[n], vhat[n], rhs_v[n], rhs_v_old[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re, start_y[n], k[n], what[n], rhs_w[n], rhs_w_old[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re*Sc, start_y[n], k[n], zhat[n], rhs_z[n], rhs_z_old[n]);
		}
	}

	return;
}

__global__
void updateKernel_mgpu(int start_y, cufftDoubleComplex *rhs_f, cufftDoubleComplex *rhs_f_old)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Update old variables to store current iteration
	rhs_f_old[idx].x = rhs_f[idx].x;
	rhs_f_old[idx].y = rhs_f[idx].y;

	// Zero out RHS arrays
	rhs_f[idx].x = 0.0;
	rhs_f[idx].y = 0.0;

	return;
}

void update(int nGPUs, int *NY_per_GPU, int *start_y, cufftDoubleComplex **rhs_u1, cufftDoubleComplex **rhs_u1_old, cufftDoubleComplex **rhs_u2, cufftDoubleComplex **rhs_u2_old, cufftDoubleComplex **rhs_u3, cufftDoubleComplex **rhs_u3_old, cufftDoubleComplex **rhs_z, cufftDoubleComplex **rhs_z_old)
{
	int n;
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(NY_per_GPU[n], TY), divUp(NZ2, TZ));

		updateKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], rhs_u1[n], rhs_u1_old[n]);
		updateKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], rhs_u2[n], rhs_u2_old[n]);
		updateKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], rhs_u3[n], rhs_u3_old[n]);
		updateKernel_mgpu<<<gridSize, blockSize>>>(start_y[n], rhs_z[n], rhs_z_old[n]);
	}

	return;
}

int main (void)
{
	// Set GPU's to use and list device properties
	int n, nGPUs;
	// Query number of devices attached to host
	cudaGetDeviceCount(&nGPUs);
	// List properties of each device
	displayDeviceProps(nGPUs);

	printf("Running multiGPU_3D_NSSwScalar_rev1 using %d GPUs on a %dx%dx%d grid.\n",nGPUs,NX,NY,NZ);
	
	// Split data according to number of GPUs
	int NX_per_GPU[nGPUs], NY_per_GPU[nGPUs], start_x[nGPUs], start_y[nGPUs];
	splitData(nGPUs, NX, NX_per_GPU, start_x);
	splitData(nGPUs, NY, NY_per_GPU, start_y);

	// Declare array of pointers to hold cuFFT plans
	cufftHandle *plan2d;
	cufftHandle *invplan2d;
	cufftHandle *plan1d;
    size_t *worksize_f, *worksize_i;
    cufftDoubleComplex **workspace;

	// Allocate memory for cuFFT plans
	// Allocate pinned memory on the host side that stores array of pointers
	cudaHostAlloc((void**)&plan2d, nGPUs*sizeof(cufftHandle), cudaHostAllocMapped);
	cudaHostAlloc((void**)&invplan2d, nGPUs*sizeof(cufftHandle), cudaHostAllocMapped);
	cudaHostAlloc((void**)&plan1d, nGPUs*sizeof(cufftHandle), cudaHostAllocMapped);
    cudaHostAlloc((void**)&worksize_f, nGPUs*sizeof(size_t *), cudaHostAllocMapped);
    cudaHostAlloc((void**)&worksize_i, nGPUs*sizeof(size_t *), cudaHostAllocMapped);
    cudaHostAlloc((void**)&workspace, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

    // Create plans for cuFFT on each GPU
    plan1dFFT(nGPUs, plan1d);
    plan2dFFT(nGPUs, NX_per_GPU, worksize_f, worksize_i, workspace, plan2d, invplan2d);

	// Allocate memory on host
	double **h_u;
	double **h_v;
	double **h_w;
	double **h_z;

	h_u = (double **)malloc(sizeof(double *)*nGPUs);
	h_v = (double **)malloc(sizeof(double *)*nGPUs);
	h_w = (double **)malloc(sizeof(double *)*nGPUs);
	h_z = (double **)malloc(sizeof(double *)*nGPUs);

	for(n=0; n<nGPUs; ++n){
		h_u[n] = (double *)malloc(sizeof(complex double)*NX_per_GPU[n]*NY*NZ2);
		h_v[n] = (double *)malloc(sizeof(complex double)*NX_per_GPU[n]*NY*NZ2);
		h_w[n] = (double *)malloc(sizeof(complex double)*NX_per_GPU[n]*NY*NZ2);
		h_z[n] = (double *)malloc(sizeof(complex double)*NX_per_GPU[n]*NY*NZ2);
	}
	
	// Declare variables
	// double time;
	double **k;

	cufftDoubleReal **u;
	cufftDoubleReal **v;
	cufftDoubleReal **w;
	cufftDoubleReal **z;

	cufftDoubleComplex **uhat;
	cufftDoubleComplex **vhat;
	cufftDoubleComplex **what;
	cufftDoubleComplex **zhat;

	cufftDoubleComplex **rhs_u;
	cufftDoubleComplex **rhs_v;
	cufftDoubleComplex **rhs_w;
	cufftDoubleComplex **rhs_z;

	cufftDoubleComplex **rhs_u_old;
	cufftDoubleComplex **rhs_v_old;
	cufftDoubleComplex **rhs_w_old;
	cufftDoubleComplex **rhs_z_old;

	cufftDoubleComplex **temp;
	cufftDoubleComplex **temp_reorder;
	cufftDoubleComplex **temp_advective;

	// Allocate pinned memory on the host side that stores array of pointers
	cudaHostAlloc((void**)&k, nGPUs*sizeof(double *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&uhat, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vhat, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&what, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&zhat, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&rhs_u, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_v, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_w, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_z, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&rhs_u_old, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_v_old, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_w_old, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_z_old, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&temp, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&temp_reorder, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&temp_advective, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	
	
	// Allocate memory for arrays
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		checkCudaErrors( cudaMalloc((void **)&k[n], sizeof(double)*NX ) );

		checkCudaErrors( cudaMalloc((void **)&uhat[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) ); 
		checkCudaErrors( cudaMalloc((void **)&vhat[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&what[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&zhat[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );

		checkCudaErrors( cudaMalloc((void **)&rhs_u[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );          
		checkCudaErrors( cudaMalloc((void **)&rhs_v[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs_w[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs_z[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );

		checkCudaErrors( cudaMalloc((void **)&rhs_u_old[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs_v_old[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs_w_old[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs_z_old[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );

		checkCudaErrors( cudaMalloc((void **)&temp[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&temp_reorder[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&temp_advective[n], sizeof(cufftDoubleComplex)*NX_per_GPU[n]*NY*NZ2) );
		
		printf("Data allocated on Device #%d\n", n);
	}

	// Set pointers for real arrays
	u = (cufftDoubleReal **)uhat;
	v = (cufftDoubleReal **)vhat;
	w = (cufftDoubleReal **)what;
	z = (cufftDoubleReal **)zhat;

	// printf("Starting Timer...\n");
	// StartTimer();

	// Launch CUDA kernel to initialize velocity field
	// initializeVelocity(nGPUs, start_x, NX_per_GPU, u, v, w);
	importVelocity(nGPUs, start_x, NX_per_GPU, h_u, h_v, h_w, u, v, w);

	// Initialize Scalar Field
	initializeScalar(nGPUs, start_x, NX_per_GPU, z);
	// importScalar(nGPUs, start_x, NX_per_GPU, h_z, z);

	// Setup wavespace domain
	initializeWaveNumbers(nGPUs, k);

	// Save Initial Data to file (t = 0)
	// Copy data to host   
	printf("Copy results to CPU memory...\n");
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
		checkCudaErrors( cudaMemcpyAsync(h_u[n], u[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(h_v[n], v[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(h_w[n], w[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(h_z[n], z[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
	}

	// Write data to file
    writeData_mgpu(nGPUs, NX_per_GPU, start_x, 0, 'u', h_u);
	writeData_mgpu(nGPUs, NX_per_GPU, start_x, 0, 'v', h_v);
	writeData_mgpu(nGPUs, NX_per_GPU, start_x, 0, 'w', h_w);
	writeData_mgpu(nGPUs, NX_per_GPU, start_x, 0, 'z', h_z);

	// Transform velocity back to fourier space for timestepping
	forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, u);
	forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, v);
	forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, w);
	forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, z);

	// Dealias the solution by truncating RHS
	deAlias(nGPUs, NY_per_GPU, start_y, k, uhat, vhat, what, zhat);

	// Synchronize GPUs before entering timestepping loop
	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
	}

	int c, euler;
	double time;
	printf("Entering time stepping loop...\n");
	// Enter time stepping loop
	for ( c = 1; c <= nt; ++c ){
		// Start iteration timer
		StartTimer();

		// Create flags to specify Euler timesteps

		if (c == 1){
			euler = 1;
		}
		else{
			euler = 0;
		}

		// Form the vorticity in Fourier space
		calcVorticity(nGPUs, NY_per_GPU, start_y, k, uhat, vhat, what, rhs_u, rhs_v, rhs_w);

		// Inverse Fourier Transform the vorticity to physical space.
		inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, rhs_u);
		inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, rhs_v);
		inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, rhs_w);

		// printf("Vorticity transformed to physical coordinates...\n");

		// Inverse transform the velocity to physical space to for advective terms
		inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, uhat);
		inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, vhat);
		inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, what);

		// Form non-linear terms in physical space
		formCrossProduct(nGPUs, NX_per_GPU, start_x, u, v, w, (cufftDoubleReal **)rhs_u, (cufftDoubleReal **)rhs_v, (cufftDoubleReal **)rhs_w);

		// Transform omegaXu from physical space to fourier space 
		forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, (cufftDoubleReal **)rhs_u);
		forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, (cufftDoubleReal **)rhs_v);
		forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, (cufftDoubleReal **)rhs_w);

		// Form advective terms in scalar equation
		formScalarAdvection(plan1d, plan2d, invplan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, temp_advective, k, u, v, w, zhat, rhs_z);
		
		// Transform the non-linear term in rhs from physical space to Fourier space for timestepping
		forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, (cufftDoubleReal **)rhs_z);

		// Transform velocity back to fourier space for timestepping
		forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, u);
		forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, v);
		forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, w);

		// Form right hand side of the N-S and scalar equations
		makeRHS(nGPUs, NY_per_GPU, start_y, k, rhs_u, rhs_v, rhs_w, rhs_z);

		// Dealias the solution by truncating RHS
		deAlias(nGPUs, NY_per_GPU, start_y, k, rhs_u, rhs_v, rhs_w, rhs_z);

		// Step the vector fields forward in time
		timestep(euler, nGPUs, NY_per_GPU, start_y, k, uhat, rhs_u, rhs_u_old, vhat, rhs_v, rhs_v_old, what, rhs_w, rhs_w_old, zhat, rhs_z, rhs_z_old);

		// Update loop variables to next timestep
		update(nGPUs, NY_per_GPU, start_y, rhs_u, rhs_u_old, rhs_v, rhs_v_old, rhs_w, rhs_w_old, rhs_z, rhs_z_old);

		// Synchronize GPUs before moving to next timestep
		for(n=0; n<nGPUs; ++n){
			cudaSetDevice(n);
			cudaDeviceSynchronize();
		}

		// Get elapsed time from Timer
		time = GetTimer();
		printf("Timestep %d complete, elapsed time: %2.2fs\n", c, time/1000);

		// Save data to file every n_save timesteps
		if ( c % n_save == 0 ){

			// Inverse Fourier Transform the velocity back to physical space for saving to file.
			inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, uhat);
			inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, vhat);
			inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, what);
			inverseTransform(plan1d, invplan2d,  nGPUs,  NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, zhat);

			// Copy data to host   
			printf( "Timestep %i Complete. . .\n", c );
			for(n=0; n<nGPUs; ++n){
				cudaSetDevice(n);
				cudaDeviceSynchronize();
				checkCudaErrors( cudaMemcpyAsync(h_u[n], u[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
				checkCudaErrors( cudaMemcpyAsync(h_v[n], v[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
				checkCudaErrors( cudaMemcpyAsync(h_w[n], w[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
				checkCudaErrors( cudaMemcpyAsync(h_z[n], z[n], sizeof(complex double)*NX_per_GPU[n]*NY*NZ2, cudaMemcpyDefault) );
			}

			// Write data to file
		    writeData_mgpu(nGPUs, NX_per_GPU, start_x, c, 'u', h_u);
			writeData_mgpu(nGPUs, NX_per_GPU, start_x, c, 'v', h_v);
			writeData_mgpu(nGPUs, NX_per_GPU, start_x, c, 'w', h_w);
			writeData_mgpu(nGPUs, NX_per_GPU, start_x, c, 'z', h_z);

			// Transform fields back to fourier space for timestepping
			forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, u);
			forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, v);
			forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, w);
			forwardTransform(plan1d, plan2d, nGPUs, NX_per_GPU, start_x, NY_per_GPU, start_y, temp, temp_reorder, z);
		}

	}

	// time = GetTimer();
	// printf("Time Elapsed: %2.2fs\n", time/1000);

	// Post-Simulation cleanup

	// Deallocate resources
	for(n = 0; n<nGPUs; ++n){
		cufftDestroy(plan1d[n]);
		cufftDestroy(plan2d[n]);
		cufftDestroy(invplan2d[n]);
	}

	// Deallocate GPU memory
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

        cudaFree(plan1d);
        cudaFree(plan2d);
        cudaFree(invplan2d);
        cudaFree(worksize_f);
        cudaFree(worksize_i);
        cudaFree(workspace);

		cudaFree(k[n]);

		cudaFree(uhat[n]);
		cudaFree(vhat[n]);
		cudaFree(what[n]);
		cudaFree(zhat[n]);

		cudaFree(rhs_u[n]);
		cudaFree(rhs_v[n]);
		cudaFree(rhs_w[n]);
		cudaFree(rhs_z[n]);

		cudaFree(rhs_u_old[n]);
		cudaFree(rhs_v_old[n]);
		cudaFree(rhs_w_old[n]);
		cudaFree(rhs_z_old[n]);

		cudaFree(temp[n]);
		cudaFree(temp_reorder[n]);
		cudaFree(temp_advective[n]);
		
	}
	
	// Deallocate pointer arrays on host memory
	cudaFreeHost(k);

	cudaFreeHost(uhat);
	cudaFreeHost(vhat);
	cudaFreeHost(what);
	cudaFreeHost(zhat);

	cudaFreeHost(rhs_u);
	cudaFreeHost(rhs_v);
	cudaFreeHost(rhs_w);
	cudaFreeHost(rhs_z);

	cudaFreeHost(rhs_u_old);
	cudaFreeHost(rhs_v_old);
	cudaFreeHost(rhs_w_old);
	cudaFreeHost(rhs_z_old);

	cudaFreeHost(temp);
	cudaFreeHost(temp_reorder);
	cudaFreeHost(temp_advective);

	cudaFreeHost(plan1d);
	cudaFreeHost(plan2d);
	cudaFreeHost(invplan2d);

	cudaDeviceReset();

	return 0;
}
