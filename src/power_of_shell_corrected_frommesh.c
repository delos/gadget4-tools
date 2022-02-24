// takes grid file and outputs its power spectrum

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <fftw3.h>

static ptrdiff_t n, n3, npad, n3pad, n1d;
static float *rho;
static double *k, *pk, R, T, rhomean;

double sinc(double x) {
  if(x > 0.1) return sin(x)/x;
  else return 1. + x*x*(-0.16666666666666666 + x*x*(0.008333333333333333 + x*x*(-0.0001984126984126984 + 2.7557319223985893e-6*x*x)));
}

double windowfun(double r) {
  double x = (r-R)/T;
  if(x>1. || x<-1.) return 0.;
  return (1 - cos(M_PI*(1.+x)))/(2.*sqrt(3*M_PI*T)*r);
}

int read_rho(char *filename) {

  struct stat st;
  stat(filename, &st);
  n3 = st.st_size/sizeof(float);
  n = (int)(pow(n3,1./3)+.5);
  npad = 2*(n/2+1);
  n3pad = n*n*npad;

  printf("reading file %s (%td elements -> n=%td)\n",filename,n3,n);
  FILE *fp = fopen(filename, "rb");

  rho = (float *)fftwf_malloc(sizeof(float)*n3pad);

  for(ptrdiff_t x = 0; x < n; x++) {
    for(ptrdiff_t y = 0; y < n; y++) {
      fread(rho+npad*(y+n*x),sizeof(float),n,fp);
    }
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

  double box  = 1.;
  double box3 = box*box*box;
  double dk   = 2.*M_PI/box;
  double d3k  = dk*dk*dk;

  double kNfac = M_PI / n;

  n1d = (n-1)/2;
  printf("computing %d P(k) values\n",n1d);
  pk = (double *)malloc(sizeof(double)*n1d);
  ptrdiff_t *pkcount= (ptrdiff_t *)malloc(sizeof(ptrdiff_t)*n1d);
  k = (double *)malloc(sizeof(double)*n1d);
  for(ptrdiff_t i=0; i<n1d; i++) {
    pk[i] = 0.;
    pkcount[i] = 0;
    k[i] = 0.;
  }
  for(ptrdiff_t x=0; x<n; x++) {
    ptrdiff_t x_ = x;
    if(x_ >= n/2) x_ -= n;
    for(ptrdiff_t y=0; y<n; y++) {
      ptrdiff_t y_ = y;
      if(y_ >= n/2) y_ -= n;
      for(ptrdiff_t z=0; z<npad/2; z++) {
        double rfloat = sqrt(x_*x_+y_*y_+z*z);
        ptrdiff_t r = (ptrdiff_t)(rfloat+0.5);
        if(r<=n1d && r>=1) {
          float dr = rhok[z+(npad/2)*(y+n*x)][0]/n3;
          float di = rhok[z+(npad/2)*(y+n*x)][1]/n3;
          double p = (double)(dr*dr+di*di) * box3;
          //double W = sinc(x_*kNfac) * sinc(y_*kNfac) * sinc(z*kNfac); // CIC kernel
          pk[r-1] += p; // / (W*W*W*W);
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
  printf("saving to %s\n",filename);
  FILE *fp = fopen(filename,"w");
  fprintf(fp,"# R=%lg\n# T=%lg\n",R,T);
  fprintf(fp,"# 0\n");
  fprintf(fp,"# %lg\n",rhomean);
  for(int i=0; i<n1d; i++) {
    fprintf(fp,"%le %le\n",k[i],pk[i]);
  }
  fclose(fp);

  return 0;
}

int clean() {
  fftwf_free(rho);
  free(pk);
  free(k);
  return 0;
}

int window() {
  printf("windowing density field between %lg and %lg\n",R-T,R+T);
  double wsum = 0.;
  double rhosum = 0.;
  for(ptrdiff_t x = 0; x < n; x++) {
    double x_ = (x + .5)/n - .5;
    for(ptrdiff_t y = 0; y < n; y++) {
      double y_ = (y + .5)/n - .5;
      for(ptrdiff_t z = 0; z < n; z++) {
        double z_ = (z + .5)/n - .5;
        double r = sqrt(x_*x_+y_*y_+z_*z_);
        double w = windowfun(r);
        rho[z+npad*(y+n*x)] *= w;
        rhosum += rho[z+npad*(y+n*x)];
        wsum += w;
      }
    }
  }
  rhomean = rhosum / wsum;
  return 0;
}

int main(int argc, char **argv)
{
  if(argc<5) {
    printf("usage: exe <density-file> <radius> <thickness> <out-file>\n");
    return 1;
  }

  R = atof(argv[2]);
  T = atof(argv[3]);

  read_rho(argv[1]);

  window();

  power();
  save(argv[4]);
  clean();

  return 0;
}

