// find nearest local maximum in density field
// author: Sten Delos

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <assert.h>
#include <cstddef>

#define IX(i) (((i)+n)%n)
#define IXD(i) (((i) < -n/2) ? ((i)+n) : (((i) > n/2) ? ((i)-n) : (i)) )

static ptrdiff_t n, n3;
static float * delta;
static time_t timer;

double find_local_max(ptrdiff_t *i, ptrdiff_t *j, ptrdiff_t *k) {
  ptrdiff_t init_i = *i, init_j = *j, init_k = *k;
  printf("%g (%td %td %td) -> ",delta[*k+n*(*j+n*(*i))],*i,*j,*k);
  fflush(stdout);
  while(1) {
    int mi = 0, mj = 0, mk = 0;
    double shift, maxshift = 0.;
    for(int a=-1; a<2; a++)
      for(int b=-1; b<2; b++)
        for(int c=-1; c<2; c++) {
          shift = delta[IX(*k+c) + n*(IX(*j+b) + n*IX(*i+a))] - delta[*k+n*(*j+n*(*i))];
          if(shift>0) {
            shift *= shift;
            shift /= a*a + b*b + c*c; // (shift per distance)^2
            if(shift > maxshift) {
              mi = a; mj = b; mk = c;
              maxshift = shift;
            }
          }
        }
    if(maxshift > 0.) {
      *i = IX(*i + mi);
      *j = IX(*j + mj);
      *k = IX(*k + mk);
    }
    else break;
  }
  printf("%g (%td %td %td) local max (t=%.0lfs)\n",delta[*k+n*(*j+n*(*i))],*i,*j,*k,difftime(time(NULL),timer));
  return sqrt(
      IXD(*i-init_i)*IXD(*i-init_i)+
      IXD(*j-init_j)*IXD(*j-init_j)+
      IXD(*k-init_k)*IXD(*k-init_k)
      );
}

double find_max(ptrdiff_t *i, ptrdiff_t *j, ptrdiff_t *k, double r) {
  if(r<0) return 0;
  ptrdiff_t init_i = *i, init_j = *j, init_k = *k;
  printf("%g (%td %td %td) -> ",delta[*k+n*(*j+n*(*i))],*i,*j,*k);
  fflush(stdout);
  int ma = 0, mb = 0, mc = 0, ri = (int)r;
  double d,maxd = 0.;
  for(int a=-ri; a<=ri; a++)
    for(int b=-ri; b<=ri; b++)
      for(int c=-ri; c<=ri; c++) {
        if(a*a+b*b+c*c <= r*r) {
          d = delta[IX(*k+c) + n*(IX(*j+b) + n*IX(*i+a))];
          if(d > maxd) {
            maxd = d;
            ma = a; mb = b; mc = c;
          }
        }
      }
  *i = IX(*i + ma);
  *j = IX(*j + mb);
  *k = IX(*k + mc);
  printf("%g (%td %td %td) regional max (t=%.0lfs)\n",delta[*k+n*(*j+n*(*i))],*i,*j,*k,difftime(time(NULL),timer));
  for(int a=-1; a<2; a++)
    for(int b=-1; b<2; b++)
      for(int c=-1; c<2; c++)
        if(delta[IX(*k+c) + n*(IX(*j+b) + n*IX(*i+a))] > delta[*k+n*(*j+n*(*i))]) {
          *i = -1;
          *j = -1;
          *k = -1;
          printf("not local max!\n");
          return -1;
        }
  return sqrt(
      IXD(*i-init_i)*IXD(*i-init_i)+
      IXD(*j-init_j)*IXD(*j-init_j)+
      IXD(*k-init_k)*IXD(*k-init_k)
      );
}

int main(int argc, char *argv[]) {
  timer = time(NULL);
  FILE *fp;

  if(argc<6) {
    printf("usage: exe <gridfile> <boxsize> <x> <y> <z> [r] [halo #, for output]\n");
    return 1;
  }

  // get grid size
  struct stat st;
  stat(argv[1], &st);
  n3 = st.st_size/sizeof(float);
  n = (int)(pow(n3,1./3)+.5);

  // get other input
  float boxsize = (float)atof(argv[2]);
  int halon = -1;
  printf("n = %d, boxsize = %.3g\n",n,boxsize);
  if(argc>7) halon = atoi(argv[7]);
  double r = -1;
  if(argc>6) r = atof(argv[6]);

  // get position
  double dr = boxsize/n;
  double xh = atof(argv[3]);
  double yh = atof(argv[4]);
  double zh = atof(argv[5]);
  printf("%lg %lg %lg Mpc\n",xh,yh,zh);
  ptrdiff_t ih[3], ihwalk[3], ihmax[3];
  ih[0] = (ptrdiff_t)(xh/dr);
  ih[1] = (ptrdiff_t)(yh/dr);
  ih[2] = (ptrdiff_t)(zh/dr);
  for(int d=0;d<3;d++) {
    ihwalk[d] = ih[d];
    ihmax[d] = ih[d];
  }

  // initialize delta field
  delta = (float *)malloc(sizeof(float)*n3);

  // load field
  fp = fopen(argv[1],"rb");
  fread(delta,sizeof(float),n3,fp);
  fclose(fp);

  // find maximum inside r
  double dista = find_max(ihmax,ihmax+1,ihmax+2,r/dr);
  if(dista >= 0) printf("maximum at distance %.2lf cells\n",dista);
  else for(int d=0;d<3;d++) ihmax[d] = -1;

  // move to peak
  double distl = find_local_max(ihwalk,ihwalk+1,ihwalk+2);
  printf("walked %.2lf cells to local maximum\n",distl);

  // save to file
  if(halon >= 0) {
    fp = fopen("halo_initial_grid.txt","a");
    fprintf(fp,"%d  %td %td %td  %td %td %td %lg  %td %td %td %lg\n",
        halon,ih[0],ih[1],ih[2],
        ihmax[0],ihmax[1],ihmax[2],dista*dr,
        ihwalk[0],ihwalk[1],ihwalk[2],distl*dr);
    fclose(fp);
    fp = fopen("halo_initial_grid_abs.txt","a");
    fprintf(fp,"%d %td %td %td %lg\n",halon,ihmax[0],ihmax[1],ihmax[2],dista*dr);
    fclose(fp);
    fp = fopen("halo_initial_grid_walk.txt","a");
    fprintf(fp,"%d %td %td %td %lg\n",halon,ihwalk[0],ihwalk[1],ihwalk[2],distl*dr);
    fclose(fp);
  }

  // clean up
  free(delta);

  return 0;
}

