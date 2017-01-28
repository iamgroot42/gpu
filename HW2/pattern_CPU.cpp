/*****************************************************************************
*
* String Pattern Matching - Serial Implementation
* 
* Reference: http://people.maths.ox.ac.uk/~gilesm/cuda/
*
*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

#define LINEWIDTH 20

void matchPattern_CPU(unsigned int *text, unsigned int *words, int *matches, int nwords, int length)
{
	unsigned int word;

	for (int l=0; l<length; l++)
	{
		for (int offset=0; offset<4; offset++)
		{
			if (offset==0)
				word = text[l];
			else
				word = (text[l]>>(8*offset)) + (text[l+1]<<(32-8*offset)); 

			for (int w=0; w<nwords; w++) 
				matches[w] += (word==words[w]);
		}
	}
}


int main(int argc, const char **argv)
{

	int length, len, nwords=32, matches[nwords];
	char *ctext, keywords[nwords][LINEWIDTH], *line;
	line = (char*) malloc(sizeof(char)*LINEWIDTH);
	unsigned int  *text,  *words;
	memset(matches, 0, sizeof(matches));


	// read in text and keywords for processing
	FILE *fp, *wfile;
	wfile = fopen("./data/keywords.txt","r");
	if (!wfile)
	{	printf("keywords.txt: File not found.\n");	exit(0);}

	int k=0, cnt = nwords;
	size_t read, linelen = LINEWIDTH;
	while((read = getline(&line, &linelen, wfile)) != -1 && cnt--)
	{
		strncpy(keywords[k], line, sizeof(line));
		keywords[k][4] = '\0';
		k++;
	}
	fclose(wfile);



	fp = fopen("./data/large.txt","r");
	if (!fp)
	{	printf("Unable to open the file.\n");	exit(0);}

	length = 0;
	while (getc(fp) != EOF) length++;
	ctext = (char *) malloc(length+4);

	rewind(fp);

	for (int l=0; l<length; l++) ctext[l] = getc(fp);
	for (int l=length; l<length+4; l++) ctext[l] = ' ';

	fclose(fp);


	// define number of words of text, and set pointers
	len  = length/4;
	text = (unsigned int *) ctext;

	// define words for matching
	words = (unsigned int *) malloc(nwords*sizeof(unsigned int));

	for (int w=0; w<nwords; w++)
	{
		words[w] = ((unsigned int) keywords[w][0])
             + ((unsigned int) keywords[w][1])*(1<<8)
             + ((unsigned int) keywords[w][2])*(1<<16)
             + ((unsigned int) keywords[w][3])*(1<<24);
	}

	// CPU execution
	const clock_t begin_time = clock();
	matchPattern_CPU(text, words, matches, nwords, len);
	float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
	printf("Time for matching keywords: %fs\n\n", runTime);

	printf("Printing Matches:\n");
	printf("Word\t  |\tNumber of Matches\n===================================\n");
	for (int i = 0; i < nwords; ++i)
		printf("%s\t  |\t%d\n", keywords[i], matches[i]);

	free(ctext);
	free(words);
}
