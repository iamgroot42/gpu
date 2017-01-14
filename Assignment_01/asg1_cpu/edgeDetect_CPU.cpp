/*****************************************************************************
*
* Prewitt Operator for Edge Detection - Serial Implementation
*
*****************************************************************************/


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <IL/il.h>
#include <IL/ilu.h>

using namespace std;

int Prewitt_x[3][3] = {
	{-1, 0, 1},
	{-1, 0, 1},
	{-1, 0, 1}
};

int Prewitt_y[3][3] = {
	{1, 1, 1},
	{0, 0, 0},
	{-1, -1, -1}
};

void saveImage(string filename, int width, int height, unsigned char * bitmap)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(width, height, 0, 1,IL_LUMINANCE, IL_UNSIGNED_BYTE, bitmap);
	iluFlipImage();
	ilEnable(IL_FILE_OVERWRITE);
	ilSave(IL_PNG, filename.c_str());
	fprintf(stderr, "Image saved as: %s\n", filename.c_str());
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


void edgeDetectCPU(int width, int height, unsigned char * bitmap, unsigned char ** edgemap)
{
	*edgemap = new unsigned char[width*height];
	int *tempmap = new int[width*height];
	
	int tl, tm, tr, ml, mm, mr, bl, bm, br;
	float pixval;
	unsigned char val;
	int grad_x, grad_y;
	int maxgrad = -1, mingrad = 50000;

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			
			if (!i || !j || i == (height-1) || j == (width-1))
				*(tempmap + width*i+j) = 0;
			else
			{
				tl = width*(i-1) + (j-1), /*top left pixel*/	tm = width*i+(j-1),	/*top middle pixel*/	tr = width*(i+1) + (j-1); /*top right pixel*/

				ml = width*(i-1) + j, /*middle left pixel*/		mm = width*i+j, /*middle middle pixel*/		mr = width*(i+1) + j; /*middle right pixel*/
				
				bl = width*(i-1) + (j+1), /*bottom left pixel*/	bm = width*i+(j+1), /*bottom middle pixel*/	br = width*(i+1) + (j+1); /*bottom right pixel*/

				

				grad_x = (Prewitt_x[0][0]*(int)bitmap[tl]) + (Prewitt_x[0][1]*(int)bitmap[tm]) + (Prewitt_x[0][2] * (int)bitmap[tr]) + 
				(Prewitt_x[1][0]*(int)bitmap[ml]) + (Prewitt_x[1][1]*(int)bitmap[mm]) + (Prewitt_x[1][2] * (int)bitmap[mr]) + 
				(Prewitt_x[2][0]*(int)bitmap[bl]) + (Prewitt_x[2][1]*(int)bitmap[bm]) + (Prewitt_x[2][2] * (int)bitmap[br]);


				grad_y = (Prewitt_y[0][0]*(int)bitmap[tl]) + (Prewitt_y[0][1]*(int)bitmap[tm]) + (Prewitt_y[0][2] * (int)bitmap[tr]) + 
				(Prewitt_y[1][0]*(int)bitmap[ml]) + (Prewitt_y[1][1]*(int)bitmap[mm]) + (Prewitt_y[1][2] * (int)bitmap[mr]) + 
				(Prewitt_y[2][0]*(int)bitmap[bl]) + (Prewitt_y[2][1]*(int)bitmap[bm]) + (Prewitt_y[2][2] * (int)bitmap[br]);

				val = (int)ceil(sqrt((grad_x*grad_x) + (grad_y*grad_y)));
				if(mingrad > val) mingrad = val;
				if(maxgrad < val) maxgrad = val;

				*(tempmap + mm) = val;
			}

		}
	}

	// Normalization between 0-256
	for (int i = 0; i < width*height; ++i)
	{
		pixval = (float)(tempmap[i]-mingrad)/(float)(maxgrad-mingrad);
		*(*edgemap + i) = (unsigned char)ceil(pixval*256.0f);
	}
}


int main(int argc, char const *argv[])
{
	string fname = "./images/wall4096.png";
	bool doSave = false; 
	
	if((argc > 1) && (*argv[1] == '-'))
	{
		string args_options (argv[1]);
		if(args_options.find("s"))
			doSave = true;
	}
	else
		fprintf(stderr, "Usage: %s [-s]\nFollowing options are supported:\n\t-s: Save output to file\n", argv[0]);

	
	ilInit();
	
	int width, height;
	unsigned char *image, *edgeImg;
	ILuint image_id = loadImage(fname.c_str(), &image, width, height);
	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}


	const clock_t begin_time = clock();
	edgeDetectCPU(width, height, image, &edgeImg);
	float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
	printf("Time for edge detection: %fs\n", runTime);

	
	if(doSave)
	{
		string imageName = "wall_edge.png";
		saveImage(imageName, width, height, edgeImg);
	}

	ilBindImage(0);
	ilDeleteImage(image_id);
	delete[] edgeImg;

	return 0;
}