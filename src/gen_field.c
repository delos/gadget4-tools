// make Gaussian random field from power spectrum
// extracted from Gadget-4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3-mpi.h>
#include <gsl/gsl_rng.h>

static int my_id, num_procs;
static ptrdiff_t n, npad, alloc_local, local_n0, local_0_start;
static double boxsize;
static unsigned int *seedtable;
static int seed;
static gsl_rng *rnd_generator;
static gsl_rng *rnd_generator_conjugate;
static int NPowerTable;

typedef struct pow_table
{
  double logk, logD;
} pow_table;
static pow_table *PowerTable;

int write_grid(char *fname, float *d) {
  FILE *fp;
  int i;
  for(i=0;i<num_procs;i++) { // avoid error-prone MPI-IO
    if(i==my_id) {
      printf("  process %d writing\n",my_id);
      if(!i) fp = fopen(fname, "wb");
      else fp = fopen(fname, "ab");
      fwrite(d,sizeof(float),local_n0*n*n,fp);
      fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return 0;
}

void unpad(float *delta) {
  ptrdiff_t i, j, k;
  for(i=0;i<local_n0;i++)
    for(j=0;j<n;j++)
      for(k=0;k<n;k++)
        delta[k+n*(j+n*i)] = delta[k+npad*(j+n*i)];
}

void initialize_rng(void) {
  rnd_generator_conjugate = gsl_rng_alloc(gsl_rng_ranlxd1);
  rnd_generator           = gsl_rng_alloc(gsl_rng_ranlxd1);
  gsl_rng_set(rnd_generator, seed);

  seedtable = (unsigned int *)malloc(n * n * sizeof(unsigned int));
  for(int i = 0; i < n / 2; i++)
  {
    for(int j = 0; j < i; j++)
      seedtable[i * n + j] = 0x7fffffff * gsl_rng_uniform(rnd_generator);

    for(int j = 0; j < i + 1; j++)
      seedtable[j * n + i] = 0x7fffffff * gsl_rng_uniform(rnd_generator);

    for(int j = 0; j < i; j++)
      seedtable[(n - 1 - i) * n + j] = 0x7fffffff * gsl_rng_uniform(rnd_generator);

    for(int j = 0; j < i + 1; j++)
      seedtable[(n - 1 - j) * n + i] = 0x7fffffff * gsl_rng_uniform(rnd_generator);

    for(int j = 0; j < i; j++)
      seedtable[i * n + (n - 1 - j)] = 0x7fffffff * gsl_rng_uniform(rnd_generator);

    for(int j = 0; j < i + 1; j++)
      seedtable[j * n + (n - 1 - i)] = 0x7fffffff * gsl_rng_uniform(rnd_generator);

    for(int j = 0; j < i; j++)
      seedtable[(n - 1 - i) * n + (n - 1 - j)] = 0x7fffffff * gsl_rng_uniform(rnd_generator);

    for(int j = 0; j < i + 1; j++)
      seedtable[(n - 1 - j) * n + (n - 1 - i)] = 0x7fffffff * gsl_rng_uniform(rnd_generator);
  }
}

int read_power_table(char *filename)
{ 
  FILE *fd;
  double k, p;

  if(!(fd = fopen(filename, "r")))
    return 1;

  NPowerTable = 0;
  do
  { 
    if(fscanf(fd, " %lg %lg ", &k, &p) == 2)
      NPowerTable++;
    else
      break;
  }
  while(1);

  fclose(fd);

  if(!my_id) {printf("found %d rows in input spectrum table\n", NPowerTable);fflush(stdout);}

  PowerTable = (pow_table *)malloc(NPowerTable * sizeof(pow_table));

  if(!(fd = fopen(filename, "r")))
    return 1;

  NPowerTable = 0;
  do
  { 
    double p;

    if(fscanf(fd, " %lg %lg ", &k, &p) == 2)
    { 
      PowerTable[NPowerTable].logk = k;
      PowerTable[NPowerTable].logD = p;
      NPowerTable++;
    }
    else
      break;
  }
  while(1);

  fclose(fd);
}

double powerspec(double k)
{
  double kold = k;

  double logk = log10(k);

  if(logk < PowerTable[0].logk || logk > PowerTable[NPowerTable - 1].logk)
    return 0;

  int binlow  = 0;
  int binhigh = NPowerTable - 1;

  while(binhigh - binlow > 1)
  {
    int binmid = (binhigh + binlow) / 2;
    if(logk < PowerTable[binmid].logk)
      binhigh = binmid;
    else
      binlow = binmid;
  }

  double dlogk = PowerTable[binhigh].logk - PowerTable[binlow].logk;

  if(dlogk == 0)
    return -1.;

  double u = (logk - PowerTable[binlow].logk) / dlogk;

  double logD = (1 - u) * PowerTable[binlow].logD + u * PowerTable[binhigh].logD;

  double Delta2 = pow(10.0, logD);

  double P = Delta2 / (4 * M_PI * kold * kold * kold);

  return P;
}

void setup_modes_in_kspace(fftwf_complex *fft_of_grid)
{
  double fac = pow(2 * M_PI / boxsize, 1.5);

  /* clear local FFT-mesh */
  memset(fft_of_grid, 0, alloc_local*2 * sizeof(float));

  double kfacx = 2.0 * M_PI / boxsize;
  double kfacy = 2.0 * M_PI / boxsize;
  double kfacz = 2.0 * M_PI / boxsize;

  for(int y = local_0_start; y < local_0_start + local_n0; y++)
    for(int z = 0; z < npad/2; z++)
    {

      // let's use the y and z plane here, because the x-column is available in full for both FFT schemes
      gsl_rng_set(rnd_generator, seedtable[y * n + z]);

      // we also create the modes for the conjugate column so that we can fulfill the reality constraint
      // by using the conjugate of the corresponding mode if needed
      int y_conj, z_conj;
      if(y > 0)
        y_conj = n - y;
      else
        y_conj = 0;

      if(z > 0)
        z_conj = n - z;
      else
        z_conj = 0;

      gsl_rng_set(rnd_generator_conjugate, seedtable[y_conj * n + z_conj]);

      double mode_ampl[n], mode_ampl_conj[n];
      double mode_phase[n], mode_phase_conj[n];

      // in this loop we precompute the modes for both columns, from low-k to high-k,
      // so that after an increase of resolution, one gets the same modes plus new ones
      for(int xoff = 0; xoff < n / 2; xoff++)
        for(int side = 0; side < 2; side++)
        {
          int x;
          if(side == 0)
            x = xoff;
          else
            x = n - 1 - xoff;

          double phase      = gsl_rng_uniform(rnd_generator) * 2 * M_PI;
          double phase_conj = gsl_rng_uniform(rnd_generator_conjugate) * 2 * M_PI;

          mode_phase[x]      = phase;
          mode_phase_conj[x] = phase_conj;

          double ampl;
          do
          {
            ampl = gsl_rng_uniform(rnd_generator);
          }
          while(ampl == 0);

          double ampl_conj;
          do
          {
            ampl_conj = gsl_rng_uniform(rnd_generator_conjugate);
          }
          while(ampl_conj == 0);

          mode_ampl[x] = ampl;

          mode_ampl_conj[x] = ampl_conj;
        }

      // now let's populate the full x-column of modes
      for(int x = 0; x < n; x++)
      {
        int xx, yy, zz;

        if(x >= (n / 2))
          xx = x - n;
        else
          xx = x;
        if(y >= (n / 2))
          yy = y - n;
        else
          yy = y;
        if(z >= (n / 2))
          zz = z - n;
        else
          zz = z;

        double kvec[3];
        kvec[0] = kfacx * xx;
        kvec[1] = kfacy * yy;
        kvec[2] = kfacz * zz;

        double kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
        double kmag  = sqrt(kmag2);

        if(fabs(kvec[0]) * boxsize / (2 * M_PI) > n / 2)
          continue;
        if(fabs(kvec[1]) * boxsize / (2 * M_PI) > n / 2)
          continue;
        if(fabs(kvec[2]) * boxsize / (2 * M_PI) > n / 2)
          continue;

        double p_of_k = powerspec(kmag);

        /* Note: kmag and p_of_k are unaffected by whether or not we use the conjugate mode */

        int conjugate_flag = 0;

        if(z == 0 || z == n / 2)
        {
          if(x > n / 2 && x < n)
            conjugate_flag = 1;
          else if(x == 0 || x == n / 2)
          {
            if(y > n / 2 && y < n)
              conjugate_flag = 1;
            else if(y == 0 || y == n / 2)
            {
              continue;
            }
          }
        }

        // determine location of conjugate mode in x column
        int x_conj;

        if(x > 0)
          x_conj = n - x;
        else
          x_conj = 0;

        if(conjugate_flag)
          p_of_k *= -log(mode_ampl_conj[x_conj]);
        else
          p_of_k *= -log(mode_ampl[x]);

        double delta = fac * sqrt(p_of_k);

        ptrdiff_t elem = (npad/2) * (n * (y - local_0_start) + x) + z;

        if(conjugate_flag)
        {
          fft_of_grid[elem][0] = delta * cos(mode_phase_conj[x_conj]);
          fft_of_grid[elem][1] = -delta * sin(mode_phase_conj[x_conj]);
        }
        else
        {
          fft_of_grid[elem][0] = delta * cos(mode_phase[x]);
          fft_of_grid[elem][1] = delta * sin(mode_phase[x]);
        }
      }
    }
}

int main(int argc, char **argv)
{
  double k2, kx, ky, kz, smth;
  double fx, fy, fz, ff;
  double fac;
  ptrdiff_t x, y, z, ip;
  fftwf_plan fft_inverse_plan;
  float *rhogrid;
  fftwf_complex *fft_of_rhogrid;
  FILE *fp;
  char outname[100];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  fftwf_mpi_init();

  if(argc < 5) {
    printf("usage: exe <power file> <box size> <grid size> <seed> [outname]\n");
    MPI_Finalize();
    return 1;
  }
  if(argc > 5) strcpy(outname,argv[5]);
  else sprintf(outname,"field");

  n = (ptrdiff_t)atoi(argv[3]);
  npad = 2*(n/2+1);

  boxsize = (double)atof(argv[2]);

  seed = atoi(argv[4]);

  if(read_power_table(argv[1])) {MPI_Finalize();return 1;};
  initialize_rng();

  alloc_local = fftwf_mpi_local_size_3d(n,n,npad/2, MPI_COMM_WORLD, &local_n0, &local_0_start);
  rhogrid = (float *)fftwf_alloc_real(alloc_local*2);
  fft_of_rhogrid = (fftwf_complex *)rhogrid;
  fft_inverse_plan = fftwf_mpi_plan_dft_c2r_3d(n,n,n,fft_of_rhogrid,rhogrid,MPI_COMM_WORLD,FFTW_ESTIMATE|FFTW_MPI_TRANSPOSED_IN);

  if(!my_id) {printf("grid initialized\n");fflush(stdout);}

  setup_modes_in_kspace(fft_of_rhogrid);

  if(!my_id) {printf("k modes prepared\n");fflush(stdout);}

  fftwf_execute(fft_inverse_plan);

  if(!my_id) {printf("inverse FFT done\n");fflush(stdout);}

  unpad(rhogrid);
  write_grid(outname,rhogrid);

  if(!my_id) {printf("wrote to %s\n",outname);fflush(stdout);}

  fftwf_free(rhogrid);
  free(seedtable);
  free(PowerTable);
  gsl_rng_free(rnd_generator);
  gsl_rng_free(rnd_generator_conjugate);

  MPI_Finalize();
  return 0;
}
