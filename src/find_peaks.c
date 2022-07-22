// find all peaks in grid
// author: Sten Delos

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cstddef>
#include <sys/stat.h>
#include <assert.h>

#define IX(i) ((i+n)%n)

int main(int argc, char **argv) {
  ptrdiff_t n, n3, i, j, k;
  float *delta;
  int *peaks;
  time_t timer;
  timer = time(NULL);
  FILE *fp;

  if(argc<2) {
    printf("usage: exe <gridfile> [troughs=0] [imin,imax]\n");
    return 1;
  }

  // get grid size
  struct stat st;
  stat(argv[1], &st);
  n3 = st.st_size/sizeof(float);
  n = (int)(pow(n3,1./3)+.5);
  printf("n = %d\n",n);
  int troughs = 0;
  if(argc>2) troughs = atoi(argv[2]);

  int imin = 0;
  int imax = n-1;
  if(argc>3) {
    char *token = strtok(argv[3],",");
    if(token) {
      imin = atoi(token);
      token = strtok(NULL,",");
      if(token) {
        imax = atoi(token);
      }
    }
  }

  // initialize delta field
  delta = (float *)malloc(sizeof(float)*n*n*(imax-imin+3));

  // load field
  fp = fopen(argv[1],"rb");
  int i0, i1;
  if(imax-imin+1 == n) {
    fread(delta,sizeof(float),n3,fp);
    i0 = 0;
    i1 = n;
  }
  else {
    if(imin == 0) {
      fread(delta+n*n,sizeof(float),n*n*(imax+2),fp);
      fseek(fp,sizeof(float)*n*n*(n-imax-3),SEEK_CUR);
      fread(delta,sizeof(float),n*n,fp);
    }
    else if(imax == n-1) {
      fread(delta+n*n*(imax-imin+2),sizeof(float),n*n,fp);
      fseek(fp,sizeof(float)*n*n*(imin-2),SEEK_CUR);
      fread(delta,sizeof(float),n*n*(imax-imin+2),fp);
    }
    else {
      fseek(fp,sizeof(float)*n*n*(imin-1),SEEK_SET);
      fread(delta,sizeof(float),n*n*(imax-imin+3),fp);
    }
    i0 = 1;
    i1 = imax-imin+2;
  }
  fclose(fp);

  // initialize peak list
  ptrdiff_t peak_lim = n3/64;
  peaks = (int *)malloc(sizeof(int)*3*peak_lim);
  ptrdiff_t peak_ct = 0;

  // find peaks
  for(i=i0; i<i1; i++) {
    for(j=0; j<n; j++) {
      for(k=0; k<n; k++) {
        if((!troughs && 0 > delta[k+n*(j+n*i)]) || (troughs && 0 < delta[k+n*(j+n*i)]))
          goto BREAK;
        for(int a=-1; a<2; a++)
          for(int b=-1; b<2; b++)
            for(int c=-1; c<2; c++)
              if((troughs && delta[IX(k+c) + n*(IX(j+b) + n*IX(i+a))] < delta[k+n*(j+n*i)])
                  || (!troughs && delta[IX(k+c) + n*(IX(j+b) + n*IX(i+a))] > delta[k+n*(j+n*i)]))
                goto BREAK;
        if(peak_ct >= peak_lim) {
          peak_lim *= 2;
          peaks = (int *)realloc(peaks,sizeof(int)*3*peak_lim);
        }
        peaks[3*peak_ct  ] = i - i0 + imin;
        peaks[3*peak_ct+1] = j;
        peaks[3*peak_ct+2] = k;
        peak_ct++;
BREAK:;
      }
    }
    printf(".");fflush(stdout);
  }
  printf("\nfound peaks\n");fflush(stdout);

  // save to file
  char outfile[300];
  sprintf(outfile,"%s_peaks.txt",argv[1]);
  if(imax-imin+1 == n) fp = fopen(outfile,"w");
  else fp = fopen(outfile,"a");
  for(i=0; i<peak_ct; i++) {
    fprintf(fp,"%d %d %d %g\n",
        peaks[3*i],peaks[3*i+1],peaks[3*i+2],
        delta[peaks[3*i+2]+n*(peaks[3*i+1]+n*(peaks[3*i] - imin + i0))]);
    fflush(fp);
  }
  fclose(fp);

  // clean up
  free(delta);
  free(peaks);

  return 0;
}

