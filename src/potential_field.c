// compute potential from a periodic delta grid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3-mpi.h>
#include <sys/stat.h>

static int my_id, num_procs;

int read_grid(char *fname, float *d, ptrdiff_t n0, ptrdiff_t l0_start, ptrdiff_t n) {
  FILE *fp;
  int i;
  for(i=0;i<num_procs;i++) { // avoid error-prone MPI-IO
    if(i==my_id) {
      printf("  process %d reading\n",my_id);
      fp = fopen(fname, "rb");
      fseek(fp,l0_start*n*n*sizeof(float),SEEK_SET);
      fread(d,sizeof(float),n0*n*n,fp);
      fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return 0;
}
int write_grid(char *fname, float *d, ptrdiff_t n0, ptrdiff_t l0_start, ptrdiff_t n) {
  FILE *fp;
  int i;
  for(i=0;i<num_procs;i++) { // avoid error-prone MPI-IO
    if(i==my_id) {
      printf("  process %d writing\n",my_id);
      if(!i) fp = fopen(fname, "wb");
      else fp = fopen(fname, "ab");
      fwrite(d,sizeof(float),n0*n*n,fp);
      fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return 0;
}

void pad(float *delta, ptrdiff_t n0, ptrdiff_t n) {
  ptrdiff_t npad = 2*(n/2+1), i, j, k;
  for(i=n0-1;i>=0;i--)
    for(j=n-1;j>=0;j--)
      for(k=n-1;k>=0;k--)
        delta[k+npad*(j+n*i)] = delta[k+n*(j+n*i)];
} 
void unpad(float *delta, ptrdiff_t n0, ptrdiff_t n) {
  ptrdiff_t npad = 2*(n/2+1), i, j, k;
  for(i=0;i<n0;i++)
    for(j=0;j<n;j++)
      for(k=0;k<n;k++)
        delta[k+n*(j+n*i)] = delta[k+npad*(j+n*i)];
}

int main(int argc, char **argv)
{
  double k2, kx, ky, kz, smth;
  double fx, fy, fz, ff;
  double fac;
  ptrdiff_t x, y, z, ip;
  ptrdiff_t n, npad, n3, n3pad;
  fftwf_plan fft_forward_plan, fft_inverse_plan;
  float *rhogrid;
  fftwf_complex *fft_of_rhogrid;
  FILE *fp;
  struct stat st;
  char outname[100];

  ptrdiff_t alloc_local, local_n0, local_0_start;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  fftwf_mpi_init();

  if(argc < 2) {
    printf("usage: exe <delta grid>\n");
    MPI_Finalize();
    return 1;
  }

  // get grid size
  stat(argv[1], &st);
  n3 = st.st_size/sizeof(float);
  n = (ptrdiff_t)(pow(n3,1./3)+.5);
  npad = 2*(n/2+1);
  n3pad = n*n*npad;

  fac = 1. / (M_PI * n3);

  // rhogrid in units of mass (per cell)
  alloc_local = fftwf_mpi_local_size_3d(n,n,npad/2, MPI_COMM_WORLD, &local_n0, &local_0_start);
  rhogrid = (float *)fftwf_alloc_real(alloc_local*2);
  fft_of_rhogrid = (fftwf_complex *)rhogrid;
  fft_forward_plan = fftwf_mpi_plan_dft_r2c_3d(n,n,n,rhogrid,fft_of_rhogrid,MPI_COMM_WORLD,FFTW_ESTIMATE);
  fft_inverse_plan = fftwf_mpi_plan_dft_c2r_3d(n,n,n,fft_of_rhogrid,rhogrid,MPI_COMM_WORLD,FFTW_ESTIMATE);

  // load field
  read_grid(argv[1],rhogrid,local_n0,local_0_start,n);
  pad(rhogrid,local_n0,n);
  if(!my_id) {printf("density grid loaded\n");fflush(stdout);}

  /* Do the FFT of the density field */

  fftwf_execute(fft_forward_plan);
  if(!my_id) {printf("forward FFT complete\n");fflush(stdout);}

  /* multiply with Green's function for the potential */

  for(x = 0; x < local_n0; x++) {
    if(x + local_0_start > n / 2) kx = x + local_0_start - n;
    else kx = x + local_0_start;
    for(y = 0; y < n; y++) {
      if(y > n / 2) ky = y - n;
      else ky = y;
      for(z = 0; z < n / 2 + 1; z++) {
        kz = z;
        k2 = kx * kx + ky * ky + kz * kz;
        if(k2 > 0) {
          smth = - fac / k2;
          ip = n * (n / 2 + 1) * x + (n / 2 + 1) * y + z;
          fft_of_rhogrid[ip][0] *= smth;
          fft_of_rhogrid[ip][1] *= smth;
        }
        else 
          fft_of_rhogrid[ip][0] = fft_of_rhogrid[ip][1] = 0.0;
      }
    }
  }
  if(!my_id) {printf("k-space multiplication complete\n");fflush(stdout);}

  /* Do the FFT to get the potential */

  fftwf_execute(fft_inverse_plan);
  if(!my_id) {printf("inverse FFT complete\n");fflush(stdout);}

  sprintf(outname,"%s_potential",argv[1]);
  unpad(rhogrid,local_n0,n);
  write_grid(outname,rhogrid,local_n0,local_0_start,n);

  fftwf_free(rhogrid);
  if(!my_id) {printf("done\n");fflush(stdout);}
  MPI_Finalize();
}
