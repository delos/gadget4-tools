// find properties of the peak density profile
// author: Sten Delos

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <sys/stat.h>
#include <fftw3-mpi.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

//#define MPIIO
#define DK (2*M_PI)
#define FINDTH 5
#define NEPS 1
#define NR 256

// density field variables
static fftwf_plan dplan, rdplan;
static ptrdiff_t n, npad, n3, n3pad, local_n0, local_0_start;
static float *delta, *delta0;
static fftwf_complex *delta_c, *delta0_c;

// peak arrays
static int *halon, *ih, *jh, *kh, npeak;

// results
static float *deltapk, *d2deltapk, *nupk, *xpk, *epk, *ppk;
static int nr, neps;
static double sigma0, sigma1, sigma2;

// MPI variables
static int my_id, num_procs;

// handle grid
int read_grid(char *fname, float *d, ptrdiff_t n0, ptrdiff_t l0_start, ptrdiff_t n) {
#ifdef MPIIO
  MPI_File srcfile;
  MPI_Datatype srctype;
  MPI_Type_contiguous(n0*n*n, MPI_FLOAT, &srctype);
  MPI_Type_commit(&srctype);
  int rc = MPI_File_open(MPI_COMM_WORLD,fname,MPI_MODE_RDONLY,MPI_INFO_NULL,&srcfile);
  if(rc) printf("error %d opening %s\n",rc,fname);
  MPI_File_set_view(srcfile,l0_start*n*n*sizeof(float),MPI_FLOAT,srctype,"native", MPI_INFO_NULL);
  MPI_File_read_all(srcfile,d,n0*n*n,MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&srcfile);
  return rc;
#else
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
#endif
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

// read peaks
int read_peaks(char *fname) {
  FILE *fp = fopen(fname, "r");
  if(!my_id) {
    printf("loaded ");
    printf(fname);
  }
  int lines = 0;
  char ch;
  char *state;
  while(!feof(fp))
  {
    ch = fgetc(fp);
    if(ch == '\n') lines++;
  }
  if(!my_id) printf(" (%d lines)\n",lines);
  rewind(fp);
  halon = (int *)malloc(sizeof(int)*lines);
  ih = (int *)malloc(sizeof(int)*lines);
  jh = (int *)malloc(sizeof(int)*lines);
  kh = (int *)malloc(sizeof(int)*lines);
  const char *term = "\t\n\r ";
  char line[256];
  int cfile = 0;
  while (fgets(line, sizeof(line), fp)) {
    char *tok = strtok_r(line, term, &state);
    halon[cfile] = strtol(tok,NULL,10);
    if(tok = strtok_r(NULL, term, &state))
      ih[cfile] = strtol(tok,NULL,10);
    else {ih[cfile] = -1;if(!my_id) printf("failed to read peak\n");}
    if(tok = strtok_r(NULL, term, &state))
      jh[cfile] = strtol(tok,NULL,10);
    else {jh[cfile] = -1;if(!my_id) printf("failed to read peak\n");}
    if(tok = strtok_r(NULL, term, &state))
      kh[cfile] = strtol(tok,NULL,10);
    else {kh[cfile] = -1;if(!my_id) printf("failed to read peak\n");}
    if(!my_id) printf("%d %d %d %d\n",halon[cfile],ih[cfile],jh[cfile],kh[cfile]);
    if(ih[cfile] >= local_0_start && ih[cfile] < local_0_start+local_n0)
      cfile++;
  }
  fclose(fp);
  return cfile;
}

// FFT kernels
double ipow(double base, int exp) {
  double result = 1;
  while (exp) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}
void get_deltapk(float *d) {
  int p;
  for(p=0;p<npeak;p++) deltapk[p] = d[kh[p]+npad*(jh[p]+n*(ih[p]-local_0_start))];
}

void do_r2c() {
  ptrdiff_t i,j,k,kx,ky,kz,idx;
  fftwf_execute(dplan);
  sigma0 = 0;
  sigma1 = 0;
  sigma2 = 0;
  for(i=0; i<local_n0; i++) {
    kx = ((int)i+local_0_start+n/2)%n-n/2;
    for(j=0; j<n; j++) {
      ky = ((int)j+n/2)%n-n/2;
      for(k=0; k<n/2+1; k++) {
        kz = k;
        idx = k+(n/2+1)*(j+n*i);
        delta0_c[idx][0] /= n3;
        delta0_c[idx][1] /= n3;
        double k2 = (kx*kx+ky*ky+kz*kz)*DK*DK; // k^2
        double d2 = delta0_c[idx][0]*delta0_c[idx][0] + delta0_c[idx][1]*delta0_c[idx][1];
        sigma0 += d2;
        sigma1 += d2*k2;
        sigma2 += d2*k2*k2;
        if(k>0&&k<n/2) {
          sigma0 += d2;
          sigma1 += d2*k2;
          sigma2 += d2*k2*k2;
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&sigma0,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&sigma1,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&sigma2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  sigma0 = sqrt(sigma0);
  sigma1 = sqrt(sigma1);
  sigma2 = sqrt(sigma2);
  if(!my_id) printf("sigma=(%lg, %lg/boxsize, %lg/boxsize^2)\n",sigma0,sigma1,sigma2);
}

void do_c2r_deriv(int a, int b) {
  ptrdiff_t i,j,k,kv[3],idx;
  for(i=0; i<local_n0; i++) {
    kv[0] = ((int)i+local_0_start+n/2)%n-n/2;
    for(j=0; j<n; j++) {
      kv[1] = ((int)j+n/2)%n-n/2;
      for(k=0; k<n/2+1; k++) {
        kv[2] = k;
        idx = k+(n/2+1)*(j+n*i);
        double mult = -kv[a]*kv[b]*DK*DK;
        delta_c[idx][0] = mult*delta0_c[idx][0];
        delta_c[idx][1] = mult*delta0_c[idx][1];
      }
    }
  }
  fftwf_execute(rdplan);
}

void get_d2deltapk() {
  int a,b,p;
  for(a=0;a<3;a++)
    for(b=0;b<3;b++) {
      do_c2r_deriv(a,b);
      for(p=0;p<npeak;p++) d2deltapk[b+3*(a+3*p)] = delta[kh[p]+npad*(jh[p]+n*(ih[p]-local_0_start))];
    }
}

void do_c2r_fun(double (*fun)(double),double r) {
  ptrdiff_t i,j,k,kv[3],idx;
  for(i=0; i<local_n0; i++) {
    kv[0] = ((int)i+local_0_start+n/2)%n-n/2;
    for(j=0; j<n; j++) {
      kv[1] = ((int)j+n/2)%n-n/2;
      for(k=0; k<n/2+1; k++) {
        kv[2] = k;
        idx = k+(n/2+1)*(j+n*i);
        double mult = (*fun)(sqrt(kv[0]*kv[0]+kv[1]*kv[1]+kv[2]*kv[2])*DK*r);
        delta_c[idx][0] = mult*delta0_c[idx][0];
        delta_c[idx][1] = mult*delta0_c[idx][1];
      }
    }
  }
  fftwf_execute(rdplan);
}

// get shape from derivative
void get_peak_shape() {
  int p,a,b;
  double lam1,lam2,lam3;
  gsl_matrix *zeta = gsl_matrix_alloc(3,3);
  gsl_vector *eval = gsl_vector_alloc(3);
  gsl_matrix *evec = gsl_matrix_alloc(3,3);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(3);
  for(p=0;p<npeak;p++) {
    for(a=0;a<3;a++)
      for(b=0;b<3;b++)
        gsl_matrix_set(zeta,a,b,d2deltapk[b+3*(a+3*p)]);
    gsl_eigen_symmv(zeta, eval, evec, w);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);
    lam1 = -gsl_vector_get(eval, 0);
    lam2 = -gsl_vector_get(eval, 1);
    lam3 = -gsl_vector_get(eval, 2);
    nupk[p] = deltapk[p]/sigma0;
    xpk[p] = (lam1+lam2+lam3)/sigma2;
    epk[p] = (lam1-lam3)/(2*(lam1+lam2+lam3));
    ppk[p] = (lam1-2*lam2+lam3)/(2*(lam1+lam2+lam3));
  }
  gsl_eigen_symmv_free(w);
}

void save_peaks() {
  FILE *fp;
  int p,i;
  if(!my_id) {
    fp = fopen("peak_parameters.txt","w");
    fprintf(fp,"#n d -del^2d e p\n");
    fclose(fp);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fp = fopen("peak_parameters.txt","a");
  for(p=0;p<npeak;p++) {
    fprintf(fp,"%d %lg %lg %lg %lg\n",
        halon[p],nupk[p]*sigma0,xpk[p]*sigma2,epk[p],ppk[p]);
  }
  fclose(fp);
}

void cleanup() {
  fftwf_free(delta);
  fftwf_free(delta0);
  fftwf_destroy_plan(dplan);
  fftwf_destroy_plan(rdplan);
  free(halon);
  free(ih);
  free(jh);
  free(kh);
  free(deltapk);
  free(d2deltapk);
  free(nupk);
  free(xpk);
  free(epk);
  free(ppk);
}

int main(int argc, char **argv) {
  ptrdiff_t alloc_local, i, j, k, d, p;
  time_t timer = time(NULL);
  FILE *fp;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  fftwf_mpi_init();

  if(!my_id) {for(i=0; i<argc; i++) {printf(argv[i]);printf(" ");}printf("\n");}
  if(argc<3) {
    if(!my_id) {
      printf("usage: exe <grid file> <peak file>\n");
      printf("  peak file has columns n, i, j, k\n");
    }
    MPI_Finalize();
    return 1;
  }

  // get grid size
  struct stat st;
  stat(argv[1], &st);
  n3 = st.st_size/sizeof(float);
  n = (int)(pow(n3,1./3)+.5);
  npad = 2*(n/2+1);
  n3pad = n*n*npad;
  if(!my_id) printf("n = %d\n",n);

  // initialize and load delta field
  alloc_local = fftwf_mpi_local_size_3d(n,n,npad/2, MPI_COMM_WORLD, &local_n0, &local_0_start);
  delta = (float *)fftwf_alloc_real(alloc_local*2);
  delta_c = (fftwf_complex *)delta;
  delta0 = (float *)fftwf_alloc_real(alloc_local*2);
  delta0_c = (fftwf_complex *)delta0; // source field for inverse transforms
  dplan = fftwf_mpi_plan_dft_r2c_3d(n,n,n,delta0,delta0_c,MPI_COMM_WORLD,FFTW_ESTIMATE);
  rdplan = fftwf_mpi_plan_dft_c2r_3d(n,n,n,delta_c,delta,MPI_COMM_WORLD,FFTW_ESTIMATE);
  read_grid(argv[1],delta0,local_n0,local_0_start,n);
  pad(delta0,local_n0,n);
  if(!my_id) printf("loaded field (t=%.0lfs)\n",difftime(time(NULL),timer));

  // get peaks
  npeak = read_peaks(argv[2]);
  deltapk = (float *)malloc(sizeof(float)*npeak);
  d2deltapk = (float *)malloc(sizeof(float)*npeak*9);
  nupk = (float *)malloc(sizeof(float)*npeak);
  xpk = (float *)malloc(sizeof(float)*npeak);
  epk = (float *)malloc(sizeof(float)*npeak);
  ppk = (float *)malloc(sizeof(float)*npeak);
  printf("process %d has %d peaks\n",my_id,npeak);

  // get delta of peaks
  get_deltapk(delta0);

  // forward FFT
  do_r2c();
  if(!my_id) printf("forward FFT done (t=%.0lfs)\n",difftime(time(NULL),timer));

  // get derivative at peak
  get_d2deltapk();
  if(!my_id) printf("peak derivatives done (t=%.0lfs)\n",difftime(time(NULL),timer));
  get_peak_shape();

  MPI_Barrier(MPI_COMM_WORLD);

  // save to halo table
  save_peaks();

  // clean up
  if(!my_id) printf("done (t=%.0lfs)\n",difftime(time(NULL),timer));
  cleanup();
  MPI_Finalize();
  return 0;
}

