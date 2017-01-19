#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <IL/il.h>
#include <IL/ilu.h>

// Reference for texture API from Nvidia's Official Documentation:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory

texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> texRef;

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

__global__ void edgeMap(unsigned char* edgemap, int width, int height){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	/* Prewitt operators' values used as it is, as storing it in an array implies
	use of local memory, which is slow to access. */

	int tl, tm, tr, ml, mr, bl, bm, br;
	unsigned char val;
	int grad_x, grad_y;

	if(i < height && j < width){
		if (!i || !j || i == (height-1) || j == (width-1))
			edgemap[width*i + j] = 0;
		else
		{	
			tl = tex2D(texRef, j-1, i-1);
			tm = tex2D(texRef, j-1, i);
			tr = tex2D(texRef, j-1, i+1);
			ml = tex2D(texRef, j, i-1);
			mr = tex2D(texRef, j, i+1);
			bl = tex2D(texRef, j+1, i-1);
			bm = tex2D(texRef, j+1, i);
			br = tex2D(texRef, j+1, i+1);
				
			grad_x = (-1*tl) + (1*tr) + (-1*ml) + (1*mr) + (-1*bl) + (1*br);

			grad_y = (1*tl) + (1*tm) + (1*tr) + (-1*bl) + (-1*bm) + (-1*br);

			val = (int)ceil(sqrt((float)((grad_x*grad_x) + (grad_y*grad_y))));

			edgemap[width*i + j] = val;
		}
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
	unsigned char *cuda_edgemap;
	int *cuda_mingrad, *cuda_maxgrad;

	ilInit();

	ILuint image_id = loadImage("./images/wall256.png", &image, width, height);

	edgemap = (unsigned char*)malloc(width * height);
	min_grad = (int*)malloc(sizeof(int));
	max_grad = (int*)malloc(sizeof(int));
	min_grad[0] = 50000;
	max_grad[0] = -1;

	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

	cudaMalloc((void**) &cuda_edgemap, width * height);
	cudaMalloc((void**) &cuda_mingrad, sizeof(int));
	cudaMalloc((void**) &cuda_maxgrad, sizeof(int));

	cudaMemset(cuda_edgemap, 0, width * height);
	cudaMemcpy(cuda_maxgrad, max_grad, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_mingrad, min_grad, sizeof(int), cudaMemcpyHostToDevice);

	int block_dim = 32;
	dim3 threadsPerBlock(block_dim, block_dim);
	dim3 numBlocks(width/block_dim, height/block_dim);

	// Texture memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaArray* cuArray; cudaMallocArray(&cuArray, &channelDesc, width, height); 
	cudaMemcpyToArray(cuArray, 0, 0, image, width*height, cudaMemcpyHostToDevice);
	texRef.addressMode[0] = cudaAddressModeWrap;
	texRef.addressMode[1] = cudaAddressModeWrap;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = false;
	cudaBindTextureToArray(texRef, cuArray, channelDesc);

	// Compute edgemap
	edgeMap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, width, height);
	// Find min and max pixel values over the edgemap
	MinMax<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, width, cuda_mingrad, cuda_maxgrad);
	cudaMemcpy(min_grad, cuda_mingrad, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(max_grad, cuda_maxgrad, sizeof(int), cudaMemcpyDeviceToHost);
	// Normalize edgemap image using overall maximum and minimum
	normalizeEdgemap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, max_grad[0], min_grad[0], width, height);
	cudaMemcpy(edgemap, cuda_edgemap, width * height, cudaMemcpyDeviceToHost);

	cudaFreeArray(cuArray);

	int i,j;
	for(i=0;i<width;i++){
		for(j=0;j<height;j++){
			printf("%d ",edgemap[i*width+j]);
		}
	}
	
	saveImage("./ohho.png", width, height, edgemap);

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
