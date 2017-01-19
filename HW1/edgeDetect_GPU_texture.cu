#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <IL/il.h>
#include <IL/ilu.h>

// Reference for texture API from Nvidia's Official Documentation:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory

__global__ void edgeMap(unsigned char* edgemap, cudaTextureObject_t bitmap , int width, int height){
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
			tl = tex2D<int>(bitmap, width*(i-1), j-1);
			tm = tex2D<int>(bitmap, width*i, j-1);
			tr = tex2D<int>(bitmap, width*(i+1), j-1);
			ml = tex2D<int>(bitmap, width*(i-1), j);
			mm = tex2D<int>(bitmap, width*i, j);
			mr = tex2D<int>(bitmap, width*(i+1), j);
			bl = tex2D<int>(bitmap, width*(i-1), j+1);
			bm = tex2D<int>(bitmap, width*i, j+1);
			br = tex2D<int>(bitmap, width*(i+1), j+1);
				
			grad_x = (Prewitt_x[0]*tl) + (Prewitt_x[1]*tm) + (Prewitt_x[2]*tr) + 
			(Prewitt_x[3]*ml) + (Prewitt_x[4]*mm) + (Prewitt_x[5]*mr) + 
			(Prewitt_x[6]*bl) + (Prewitt_x[7]*bm) + (Prewitt_x[8] * br);

			grad_y = (Prewitt_y[0]*tl) + (Prewitt_y[1]*tm) + (Prewitt_y[2]*tr) + 
			(Prewitt_y[3]*ml) + (Prewitt_y[4]*mm) + (Prewitt_y[5] * mr) + 
			(Prewitt_y[6]*bl) + (Prewitt_y[7]*bm) + (Prewitt_y[8] * br);

			val = (int)ceil(sqrt((float)((grad_x*grad_x) + (grad_y*grad_y))));

			edgemap[mm] = val;
		}
	}
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
	unsigned char *cuda_edgemap;

	ilInit();

	ILuint image_id = loadImage("./images/wall256.png", &image, width, height);

	edgemap = (unsigned char*)malloc(width * height);

	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

	cudaMalloc((void**) &cuda_edgemap, width * height);

	cudaMemset(cuda_edgemap, 0, width * height);

	dim3 threadsPerBlock(width/32, height/32);
	dim3 numBlocks(32, 32);

	// Texture memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray* cuda_bitmap;
	cudaMallocArray(&cuda_bitmap, &channelDesc, width, height);
	cudaMemcpyToArray(cuda_bitmap, 0, 0, image, width * height, cudaMemcpyHostToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuda_bitmap;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	// Compute edgemap
	edgeMap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, texObj, width, height);
	// Find min and max pixel values over the edgemap
	// TODO
	// Find min and max pixel values over the edgemap
	normalizeEdgemap<<<numBlocks, threadsPerBlock>>>(cuda_edgemap, maxgrad, mingrad, width, height);
	cudaMemcpy(edgemap, cuda_edgemap, width * height, cudaMemcpyDeviceToHost);

	cudaDestroyTextureObject(texObj);
	cudaFreeArray(cuda_bitmap);

	cudaFree(cuda_edgemap);

	free(edgemap);

	saveImage("./ohho.png", width, height, edgemap);

	ilBindImage(0);
	ilDeleteImage(image_id);

	return 0;
}
