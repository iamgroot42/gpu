// iamgroot42

// Code for reading files and loading into memory used from the CPU template provided with the assignment

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

#define LINEWIDTH 20
#define KEYWORD 32
#define CHUNKSIZE 4


__global__ void matchPattern(unsigned int *text, unsigned int *words, int *matches, int length){

	__shared__ unsigned int shared_words[CHUNKSIZE+1];
	__shared__ unsigned int keywords[KEYWORD];
	__shared__  int frequencies[KEYWORD];
	unsigned int word, next_word, offset1, offset2, offset3;

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int match_count = 0;

	if(i<length){
		// Load keywords, initialize blockwise frequencies
		if(threadIdx.x==0){
			keywords[threadIdx.y] = words[threadIdx.y];
			frequencies[threadIdx.y] = 0;
		}

		// Load text
		if(threadIdx.y==0){
			shared_words[threadIdx.x] = text[i];
			if(threadIdx.x+1==CHUNKSIZE){
				shared_words[CHUNKSIZE] = text[i+1];
			}
		}
		__syncthreads();

		// Matching logic
		word = shared_words[threadIdx.x];
		next_word = shared_words[threadIdx.x+1];
		offset1 = (word>>8) + (next_word<<24);
		offset2 = (word>>16) + (next_word<<16);
		offset3 = (word>>24) + (next_word<<8);
		match_count += (word==keywords[threadIdx.y]);
		match_count += (offset1==keywords[threadIdx.y]);
		match_count += (offset2==keywords[threadIdx.y]);
		match_count += (offset3==keywords[threadIdx.y]);

		// Increment shared counters
		atomicAdd(&frequencies[threadIdx.y], match_count);
		__syncthreads();

		// Incrememnt global histogram
		if(threadIdx.x == 0){
			atomicAdd(&matches[threadIdx.y], frequencies[threadIdx.y]);
		}
	}
}

int main(){

	int length, len, matches[KEYWORD];
	int* cuda_matches;
	char *ctext, keywords[KEYWORD][LINEWIDTH], *line;
	cudaMallocHost(&line, sizeof(char)*LINEWIDTH);
	unsigned int *text,  *words;
	unsigned int *cuda_text, *cuda_words;
	memset(matches, -1, sizeof(matches));

	// read in text and keywords for processing
	FILE *fp, *wfile;
	wfile = fopen("./data/keywords.txt","r");
	if (!wfile){ printf("keywords.txt: File not found.\n");	exit(0);}

	int k=0, cnt = KEYWORD;
	size_t read, linelen = LINEWIDTH;
	while((read = getline(&line, &linelen, wfile)) != -1 && cnt--){
		strncpy(keywords[k], line, sizeof(line));
		keywords[k][4] = '\0';
		k++;
	}
	fclose(wfile);

	fp = fopen("./data/large.txt","r");
	if (!fp){ printf("Unable to open the file.\n");	exit(0);}

	length = 0;
	while (getc(fp) != EOF) length++;
	cudaMallocHost(&ctext, length+4);

	rewind(fp);

	for (int l=0; l<length; l++) ctext[l] = getc(fp);
	for (int l=length; l<length+4; l++) ctext[l] = ' ';

	fclose(fp);

	// define number of words of text, and set pointers
	len  = length/4;
	text = (unsigned int *) ctext;

	// define words for matching
	cudaMallocHost(&words, KEYWORD*sizeof(unsigned int));

	for (int w=0; w<KEYWORD; w++){
		words[w] = ((unsigned int) keywords[w][0])
			+ ((unsigned int) keywords[w][1])*(1<<8)
			+ ((unsigned int) keywords[w][2])*(1<<16)
			+ ((unsigned int) keywords[w][3])*(1<<24);
	}

	cudaMalloc((void**) &cuda_text, length+4);
	cudaMalloc((void**) &cuda_words, KEYWORD*sizeof(unsigned int));
	cudaMalloc((void**) &cuda_matches, sizeof(matches));

	cudaMemcpy(cuda_text, text, length+4, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_words, words, KEYWORD*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemset(cuda_matches, 0, sizeof(matches));
	cudaMemset(cuda_matches, 0, sizeof(matches));

	dim3 threadsPerBlock(CHUNKSIZE, KEYWORD);
	dim3 numBlocks(len/CHUNKSIZE);

	matchPattern<<<numBlocks, threadsPerBlock>>>(cuda_text, cuda_words, cuda_matches, len);

	cudaMemcpy(matches, cuda_matches, sizeof(matches), cudaMemcpyDeviceToHost);

	printf("Printing Matches:\n");
	printf("Word\t  |\tNumber of Matches\n===================================\n");
	for (int i = 0; i < KEYWORD; ++i)
		printf("%s\t  |\t%d\n", keywords[i], matches[i]);

	cudaFree(ctext);
	cudaFree(words);
	cudaFree(line);
	cudaFree(cuda_text);
	cudaFree(cuda_words);
	cudaFree(cuda_matches);

	return 0;
}
