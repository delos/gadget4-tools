GSL_INCL   =  -I$(GSL_HOME)/include
GSL_LIBS   =  -L$(GSL_HOME)/lib  -lgsl -lgslcblas
FFTW_INCL  =  -I$(FFTW_HOME)/include
FFTW_LIBS  =  -L$(FFTW_HOME)/lib  -lfftw3f
HDF5_INCL  =  -I$(HDF5_HOME)/include
HDF5_LIBS  =  -L$(HDF5_HOME)/lib

all: powers fields

powers: power power_of_shell_corrected power_of_shell_corrected_frommesh power_of_sphere_frommesh

fields: gen_field potential_field

power_of_shell_corrected: ./src/power_of_shell_corrected.c
	g++ $(FFTW_INCL) $(FFTW_LIBS) -o ./bin/power_of_shell_corrected ./src/power_of_shell_corrected.c

power_of_shell_corrected_frommesh: ./src/power_of_shell_corrected_frommesh.c
	g++ $(FFTW_INCL) $(FFTW_LIBS) -o ./bin/power_of_shell_corrected_frommesh ./src/power_of_shell_corrected_frommesh.c

power_of_sphere_frommesh: ./src/power_of_sphere_frommesh.c
	g++ $(FFTW_INCL) $(FFTW_LIBS) -o ./bin/power_of_sphere_frommesh ./src/power_of_sphere_frommesh.c

power: ./src/power.c
	g++ $(FFTW_INCL) $(FFTW_LIBS) -o ./bin/power ./src/power.c

potential_field: ./src/potential_field.c
	mpicc $(FFTW_INCL) $(FFTW_LIBS) -lfftw3f_mpi -lm -o ./bin/potential_field ./src/potential_field.c

gen_field: ./src/gen_field.c
	mpicc $(FFTW_INCL) $(FFTW_LIBS) -lfftw3f_mpi $(GSL_INCL) $(GSL_LIBS) -lm -o ./bin/gen_field ./src/gen_field.c

