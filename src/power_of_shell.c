// takes grid file and outputs its power spectrum

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <fftw3.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

static ptrdiff_t n, n3, npad, n3pad, n1d;
static float *rho;
static double *k, *pk, R, T;

int read(char *filename) {

  struct stat st;
  stat(filename, &st);
  n3 = st.st_size/sizeof(float);
  n = (int)(pow(n3,1./3)+.5);
  npad = 2*(n/2+1);
  n3pad = n*n*npad;

  printf("reading file %s (n=%ld)\n",filename,n);
  FILE *fp = fopen(filename, "rb");

  rho = (float *)fftwf_malloc(sizeof(float)*n3pad);

  for(ptrdiff_t x = 0; x < n; x++)
    for(ptrdiff_t y = 0; y < n; y++) {
      fread(rho+npad*(y+n*x),sizeof(float),n,fp);
    }
  fclose(fp);

  return 0;
}

int power() {
  printf("executing FFT\n");
  fftwf_plan dplan = fftwf_plan_dft_r2c_3d(n,n,n,rho,(fftwf_complex *)rho,FFTW_ESTIMATE);
  fftwf_execute(dplan);
  fftwf_destroy_plan(dplan);
  fftwf_complex *rhok = (fftwf_complex *)rho;

  double dk  = 2.*M_PI; // box size = 1
  double d3k = dk*dk*dk;

  n1d = (n-1)/2;
  printf("computing %d P(k) values\n",n1d);
  pk = (double *)malloc(sizeof(double)*n1d);
  ptrdiff_t *pkcount= (ptrdiff_t *)malloc(sizeof(ptrdiff_t)*n1d);
  double *k = (double *)malloc(sizeof(double)*n1d);
  for(ptrdiff_t i=0; i<n1d; i++) {
    pk[i] = 0.;
    pkcount[i] = 0;
    k[i] = 0.;
  }
  for(ptrdiff_t x=0; x<n; x++) {
    for(ptrdiff_t y=0; y<n; y++) {
      for(ptrdiff_t z=0; z<npad/2; z++) {
        ptrdiff_t x_ = MIN(x,n-x);
        ptrdiff_t y_ = MIN(y,n-y);
        double rfloat = sqrt(x_*x_+y_*y_+z*z);
        ptrdiff_t r = (ptrdiff_t)(rfloat+0.5);
        if(r<=n1d && r>=1) {
          float dr = rhok[z+(npad/2)*(y+n*x)][0]/n3;
          float di = rhok[z+(npad/2)*(y+n*x)][1]/n3;
          pk[r-1] += (double)(dr*dr+di*di)/d3k;
          pkcount[r-1]++;
          k[r-1] += (rfloat-r)*dk;
        }
      }
    }
  }
  
  for(ptrdiff_t i=0; i<n1d; i++) {
    if(pkcount[i]>0) {
      pk[i] /= pkcount[i];
      k[i] /= pkcount[i];
    }
    k[i] += (i+1.)*dk;
  }
  free(pkcount);

  return 0;

}

int save(char *filename) {
	FILE *fp = fopen(filename,"w");
  fprintf(fp,"# R=%g\n# r=%g\n",R,T);
	for(int i=0; i<n1d; i++)
		fprintf(fp,"%lg %lg\n",k[i],pk[i]);
	fclose(fp);

	return 0;
}

int clean() {
	fftwf_free(rho);
	free(pk);
	free(k);
	return 0;
}

double windowfun(double r) {
  double x = (r-R)/T;
  if(x>1. || x<-1.) return 0.;
	return 0.5 * ( 1 - cos(M_PI*(x+1.)) );
}

double windowmean() {
  return 4*M_PI*R*R*T + (4*(-6 + M_PI*M_PI)*T*T*T)/(3.*M_PI);
}

double windowrms() {
	return (M_PI*T*(6*R*R + (2 - 15/(M_PI*M_PI))*T*T))/2.;
}

int window() {
  double rms = windowrms();
	for(ptrdiff_t x = 0; x < n; x++) {
		double x_ = (x + .5)/n - .5;
		for(ptrdiff_t y = 0; y < n; y++) {
			double y_ = (y + .5)/n - .5;
			for(ptrdiff_t z = 0; z < n; y++) {
				double z_ = (z + .5)/n - .5;
				double r = sqrt(x_*x_+y_*y_+z_*z_);
				rho[z+npad*(y+n*x)] *= windowfun(r) / rms;
			}
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	if(argc<5) {
		printf("usage: exe <file> <radius> <thickness> <out-file>\n");
		return 1;
	}

	read(argv[1]);
	R = atof(argv[2]);
	T = atof(argv[3]);

	window();

	power();
	save(argv[4]);
	clean();

	return 0;
}

