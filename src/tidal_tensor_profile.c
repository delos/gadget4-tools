// find profile of the tidal tensor -d_i d_j phi, where delta = -del^2 phi
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
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sort_vector.h>

//#define MPIIO
#define DK (2*M_PI)
#define NR 256

// density field variables
static fftwf_plan dplan, rdplan;
static ptrdiff_t n, npad, n3, n3pad, local_n0, local_0_start;
static float *delta, *delta0;
static fftwf_complex *delta_c, *delta0_c;

// peak arrays
static int *halon, *ih, *jh, *kh, npeak;

// results
static float *rbin, *Tr;
static int nr;

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

// indexing
void a_b_from_ab(int ab,int *a, int *b) {
  switch(ab) {
    case 0: *a=0; *b=0; break;
    case 1: *a=0; *b=1; break;
    case 2: *a=0; *b=2; break;
    case 3: *a=1; *b=1; break;
    case 4: *a=1; *b=2; break;
    case 5: *a=2; *b=2; break;
  }
}
int ab_from_a_b(int a,int b) {
  if(a==0) return b;
  if(b==0) return a;
  if(a==1) return b+2;
  if(b==1) return a+2;
  return 5;
}

void prepare_r(double rmin, double rmax, int log) {
  ptrdiff_t i;
  if(log)
    for(i=0;i<nr;i++)
      rbin[i] = rmin*pow(rmax/rmin,i/(nr-1.));
  else
    for(i=0;i<nr;i++)
      rbin[i] = rmin+(rmax-rmin)*i/(nr-1.);
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
double tophat(double y) {
  if(y<=0.3) return 1. - y*y/10. + ipow(y,4)/280. - ipow(y,6)/15120. + ipow(y,8)/1330560. - ipow(y,10)/172972800.;
  else return 3./ipow(y,3)*(sin(y)-y*cos(y));
}

void do_r2c() {
  ptrdiff_t i,j,k,kx,ky,kz,idx;
  fftwf_execute(dplan);
  for(i=0; i<local_n0; i++) {
    kx = ((int)i+local_0_start+n/2)%n-n/2;
    for(j=0; j<n; j++) {
      ky = ((int)j+n/2)%n-n/2;
      for(k=0; k<n/2+1; k++) {
        kz = k;
        idx = k+(n/2+1)*(j+n*i);
        delta0_c[idx][0] /= n3;
        delta0_c[idx][1] /= n3;
      }
    }
  }
}

void do_c2r_tidal(int a,int b,double r) {
  ptrdiff_t i,j,k,kv[3],idx;
  for(i=0; i<local_n0; i++) {
    kv[0] = ((int)i+local_0_start+n/2)%n-n/2;
    for(j=0; j<n; j++) {
      kv[1] = ((int)j+n/2)%n-n/2;
      for(k=0; k<n/2+1; k++) {
        kv[2] = k;
        idx = k+(n/2+1)*(j+n*i);
        double kmag = sqrt(kv[0]*kv[0]+kv[1]*kv[1]+kv[2]*kv[2]);
        double mult;
        if(kmag>0) mult = tophat(kmag*DK*r) * kv[a] * kv[b] / (kmag * kmag);
        else mult = 1;
        delta_c[idx][0] = mult*delta0_c[idx][0];
        delta_c[idx][1] = mult*delta0_c[idx][1];
      }
    }
  }
  fftwf_execute(rdplan);
}

void get_tidal(int ab) {
  ptrdiff_t i,p;
  int a,b;
  a_b_from_ab(ab,&a,&b);
  for(i=0;i<nr;i++) {
    do_c2r_tidal(a,b,rbin[i]);
    for(p=0;p<npeak;p++) Tr[i+nr*(ab+6*p)] = delta[kh[p]+npad*(jh[p]+n*(ih[p]-local_0_start))];
    if(!my_id) printf("  computing T_{%d,%d}(%g)\n",a,b,rbin[i]);
  }
}

void save_profiles() {
  char name[256];
  FILE *fp;
  int p,i,a,b;
  gsl_matrix *T = gsl_matrix_alloc(3,3);
  gsl_vector *val = gsl_vector_alloc(3);
  gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc(3);
  for(p=0;p<npeak;p++) {
    sprintf(name,"tidal_tensor_profile_%04d.txt",halon[p]);
    fp = fopen(name,"w");
    fprintf(fp,"# %d %d %d %d\n",halon[p],ih[p],jh[p],kh[p]);
    for(i=0;i<nr;i++) {
      for(a=0; a<3; a++)
        for(b=0; b<3; b++)
          gsl_matrix_set(T,a,b,Tr[i+nr*(ab_from_a_b(a,b)+6*p)]);
      gsl_eigen_symm(T, val, w);
      gsl_sort_vector(val);
      double Td = gsl_vector_get(val,0) + gsl_vector_get(val,1) + gsl_vector_get(val,2);
      double Te = (gsl_vector_get(val,2) - gsl_vector_get(val,0)) / (2*Td);
      double Tp = (gsl_vector_get(val,2) + gsl_vector_get(val,0) - 2*gsl_vector_get(val,1)) / (2*Td);
      fprintf(fp,"%lg %lg %lg %lg\n",rbin[i],Td,Te,Tp);
    }
    fclose(fp);
  }
  gsl_eigen_symm_free(w);
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
  free(rbin);
  free(Tr);
}

int main(int argc, char **argv) {
  ptrdiff_t alloc_local, i, j, k, d, p;
  int ab;
  time_t timer = time(NULL);
  FILE *fp;
  nr = NR;

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
  rbin = (float *)malloc(sizeof(float)*nr);
  Tr = (float *)malloc(sizeof(float)*npeak*nr*6);
  printf("process %d has %d peaks\n",my_id,npeak);

  // forward FFT
  do_r2c();
  if(!my_id) printf("forward FFT done (t=%.0lfs)\n",difftime(time(NULL),timer));

  // get density profile
  prepare_r(1./n,.25,1);
  for(ab=0; ab<6; ab++) {
    get_tidal(ab);
    if(!my_id) printf("T_%d profile done (t=%.0lfs)\n",ab,difftime(time(NULL),timer));
  }

  // save to profile table
  save_profiles();

  // clean up
  if(!my_id) printf("done (t=%.0lfs)\n",difftime(time(NULL),timer));
  cleanup();
  MPI_Finalize();
  return 0;
}

