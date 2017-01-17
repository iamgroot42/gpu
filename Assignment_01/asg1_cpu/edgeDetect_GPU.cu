#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <IL/il.h>
#include <IL/ilu.h>


__global__ void edgeMap(unsigned char* edgemap, unsigned char* bitmap, int width, int height){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	int Prewitt_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1};
	int Prewitt_y[9] = { 1, 1, 1, 0, 0, 0, -1, -1, -1};

	int tl, tm, tr, ml, mm, mr, bl, bm, br;
	unsigned char val;
	int grad_x, grad_y;

	if(i < height && j < width){
		if (!i || !j || i == (height-1) || j == (width-1))
			edgemap[width*i + j] = 0;
		else
		{
			tl = width*(i-1) + (j-1), tm = width*i+(j-1), tr = width*(i+1) + (j-1); 
			ml = width*(i-1) + j, mm = width*i+j, mr = width*(i+1) + j; 
			bl = width*(i-1) + (j+1), bm = width*i+(j+1), br = width*(i+1) + (j+1); 
				
			grad_x = (Prewitt_x[0]*(int)bitmap[tl]) + (Prewitt_x[1]*(int)bitmap[tm]) + (Prewitt_x[2] * (int)bitmap[tr]) + 
			(Prewitt_x[3]*(int)bitmap[ml]) + (Prewitt_x[4]*(int)bitmap[mm]) + (Prewitt_x[5] * (int)bitmap[mr]) + 
			(Prewitt_x[6]*(int)bitmap[bl]) + (Prewitt_x[7]*(int)bitmap[bm]) + (Prewitt_x[8] * (int)bitmap[br]);

			grad_y = (Prewitt_y[0]*(int)bitmap[tl]) + (Prewitt_y[1]*(int)bitmap[tm]) + (Prewitt_y[2] * (int)bitmap[tr]) + 
			(Prewitt_y[3]*(int)bitmap[ml]) + (Prewitt_y[4]*(int)bitmap[mm]) + (Prewitt_y[5] * (int)bitmap[mr]) + 
			(Prewitt_y[6]*(int)bitmap[bl]) + (Prewitt_y[7]*(int)bitmap[bm]) + (Prewitt_y[8] * (int)bitmap[br]);

			val = (int)ceil(sqrt((float)((grad_x*grad_x) + (grad_y*grad_y))));

			edgemap[mm] = val;
		}
	}
}

__global__ void minmaxReduce(unsigned char* flat_edgemap, unsigned char* mingrad, unsigned char* maxgrad){
	extern __shared__ int reduce_mem_min[], reduce_mem_max[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	reduce_mem_min[threadIdx.x] = flat_edgemap[i];
	reduce_mem_max[threadIdx.x] = flat_edgemap[i];
	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>0; s/=2) {
		if (threadIdx.x < s) {
			reduce_mem_max[threadIdx.x] = max(reduce_mem_max[threadIdx.x], reduce_mem_max[threadIdx.x + s]);
			reduce_mem_min[threadIdx.x] = min(reduce_mem_min[threadIdx.x], reduce_mem_min[threadIdx.x + s]);
		}
		__syncthreads();
	}
	atomicMax(maxgrad, reduce_mem_max[0]);
	atomicMin(mingrad, reduce_mem_min[0]);
}

__global__ void normalizeEdgemap(unsigned char* edgemap, int maxgrad, int mingrad, int width, int height){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	float pixval;

	if(i < height && j < width){
		pixval = (float)(edgemap[width*i + j] - mingrad)/(float)(maxgrad - mingrad);
		edgemap[width*i + j] = (unsigned char)ceil(pixval*256.0f);
	}
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
	unsigned char min_grad, max_grad;
	unsigned char *cuda_img, *cuda_edgemap;
	unsigned char *cuda_mingrad, *cuda_maxgrad;

	ilInit();

	ILuint image_id = loadImage("./images/wall256.png", &image, width, height);

	edgemap = (unsigned char*)malloc(width * height);

	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

	cudaMalloc((void**) &cuda_img, width * height);
	cudaMalloc((void**) &cuda_edgemap, width * height);
	cudaMalloc((void**) &cuda_mingrad, 1);
	cudaMalloc((void**) &cuda_maxgrad, 1);

	cudaMemcpy(cuda_img, image, width * height, cudaMemcpyHostToDevice);
	cudaMemset(cuda_edgemap, 0, width * height);
	cudaMemset(cuda_mingrad, 0, 1);
	cudaMemset(cuda_maxgrad, 0, 1);

	int block_dim = 32;
	dim3 threadsPerBlock(width/block_dim, height/block_dim);
	dim3 numBlocks(block_dim, block_dim);

	dim3 threadsPerBlock_red((width*height)/(block_dim*block_dim));
	dim3 numBlocks_red(block_dim* block_dim);


	// Compute edgemap
	edgeMap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, cuda_img, width, height);
	// Find min and max pixel values over the edgemap
	minmaxReduce<<<numBlocks_red, threadsPerBlock_red>>>(cuda_edgemap, cuda_mingrad, cuda_maxgrad);
	// Find min and max pixel values over the edgemap
	// normalizeEdgemap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, maxgrad, mingrad, width, height);
	cudaMemcpy(edgemap, cuda_edgemap, width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(&min_grad, cuda_mingrad, 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&max_grad, cuda_maxgrad, 1, cudaMemcpyDeviceToHost);

	cudaFree(cuda_img);
	cudaFree(cuda_edgemap);

	printf("Max: %d\nMin: %d\n",max_grad,min_grad);
	int i,j;
	int maxx=99999,minn=-1;
	for(i=0;i<width;i++){
		for(j=0;j<height;j++){
			if(edgemap[i*width + j]<minn) minn=edgemap[i*width + j];
			if(edgemap[i*width + j]<maxx) maxx=edgemap[i*width + j];
		}
	}
	printf("Max: %d\nMin: %d\n",maxx,minn);
	free(edgemap);

	saveImage("./ohho.png", width, height, edgemap);

	ilBindImage(0);
	ilDeleteImage(image_id);

	return 0;
}
