#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <IL/il.h>
#include <IL/ilu.h>


__global__ void MinMax(unsigned char* edgemap, int width, int* mingrad, int* maxgrad)
{
	__shared__ int blockmin;
	__shared__ int blockmax;

	if(threadIdx.x == 0 && threadIdx.y == 0){
		blockmin = 50000;
		blockmax = -1;
	}
	__syncthreads();

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	atomicMax(&blockmax, edgemap[i * width + j]);
	atomicMin(&blockmin, edgemap[i * width + j]);
	__syncthreads();

	if(threadIdx.x == 0 && threadIdx.y == 0){
		atomicMax(maxgrad, blockmax);
		atomicMin(mingrad, blockmin);
	}
}

__global__ void edgeMap(unsigned char* edgemap, unsigned char* bitmap, int width, int height){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	/* Prewitt operators' values used as it is, as storing it in an array implies
	use of local memory, which is slow to access. */

	int tl, tm, tr, ml, mr, bl, bm, br;
	unsigned char val;
	int grad_x, grad_y;

	if (!i || !j || i == (height-1) || j == (width-1))
		edgemap[width*i + j] = 0;
	else
	{
		tl = width*(i-1) + (j-1), tm = width*i+(j-1), tr = width*(i+1) + (j-1); 
		ml = width*(i-1) + j, mr = width*(i+1) + j; 
		bl = width*(i-1) + (j+1), bm = width*i+(j+1), br = width*(i+1) + (j+1); 
				
		grad_x = (-1*(int)bitmap[tl]) + (1*(int)bitmap[tr]) + 
		(-1*(int)bitmap[ml]) + (1*(int)bitmap[mr]) + 
		(-1*(int)bitmap[bl]) + (1*(int)bitmap[br]);

		grad_y = (1*(int)bitmap[tl]) + (1*(int)bitmap[tm]) + (1*(int)bitmap[tr]) + 
		(-1*(int)bitmap[bl]) + (-1*(int)bitmap[bm]) + (-1*(int)bitmap[br]);

		val = (int)ceil(sqrt((float)((grad_x*grad_x) + (grad_y*grad_y))));

		edgemap[width*i+j] = val;
	}
}

__global__ void normalizeEdgemap(unsigned char* edgemap, int maxgrad, int mingrad, int width, int height){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	float pixval;

	pixval = (float)(edgemap[width*i + j] - mingrad)/(float)(maxgrad - mingrad);
	edgemap[width*i + j] = (unsigned char)ceil(pixval*256.0f);	
}

void saveImage(const char* filename, int width, int height, unsigned char * bitmap)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(width, height, 0, 1,IL_LUMINANCE, IL_UNSIGNED_BYTE, bitmap);
	iluFlipImage();
	ilEnable(IL_FILE_OVERWRITE);
	ilSave(IL_PNG, filename);
	fprintf(stderr, "Image saved as: %s\n", filename);
}

ILuint loadImage(const char *filename, unsigned char ** bitmap, int &width, int &height)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ILboolean success = ilLoadImage(filename);
	if (!success) return 0;

	width = ilGetInteger(IL_IMAGE_WIDTH);
	height = ilGetInteger(IL_IMAGE_HEIGHT);
	printf("Width: %d\t Height: %d\n", width, height);
	*bitmap = ilGetData();
	return imageID;
}

int main()
{
	int width, height;

	unsigned char *image, *edgemap;
	int *min_grad, *max_grad;
	unsigned char *cuda_img, *cuda_edgemap;
	int *cuda_mingrad, *cuda_maxgrad;

	ilInit();

	ILuint image_id = loadImage("./images/wall256.png", &image, width, height);

	edgemap = (unsigned char*)malloc(width * height);
	min_grad = (int*)malloc(sizeof(int));
	max_grad = (int*)malloc(sizeof(int));
	min_grad[0] = 50000;
	max_grad[0] = -1;

	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

	cudaMalloc((void**) &cuda_img, width * height);
	cudaMalloc((void**) &cuda_edgemap, width * height);
	cudaMalloc((void**) &cuda_mingrad, sizeof(int));
	cudaMalloc((void**) &cuda_maxgrad, sizeof(int));

	cudaMemcpy(cuda_img, image, width * height, cudaMemcpyHostToDevice);
	cudaMemset(cuda_edgemap, 0, width * height);
	cudaMemcpy(cuda_maxgrad, max_grad, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_mingrad, min_grad, sizeof(int), cudaMemcpyHostToDevice);

	int block_dim = 32;
	dim3 threadsPerBlock(block_dim, block_dim);
	dim3 numBlocks(width/block_dim, height/block_dim);

	// Compute edgemap
	edgeMap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, cuda_img, width, height);
	// Find min and max pixel values over the edgemap
	MinMax<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, width, cuda_mingrad, cuda_maxgrad);
	cudaMemcpy(min_grad, cuda_mingrad, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(max_grad, cuda_maxgrad, sizeof(int), cudaMemcpyDeviceToHost);
	// Normalize edgemap image using overall maximum and minimum
	normalizeEdgemap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, max_grad[0], min_grad[0], width, height);
	cudaMemcpy(edgemap, cuda_edgemap, width * height, cudaMemcpyDeviceToHost);

	saveImage("./ohho.png", width, height, edgemap);
	
	cudaFree(cuda_img);
	cudaFree(cuda_edgemap);
	cudaFree(cuda_mingrad);
	cudaFree(cuda_maxgrad);

	free(edgemap);
	free(max_grad);
	free(min_grad);

	ilBindImage(0);
	ilDeleteImage(image_id);

	return 0;
}
