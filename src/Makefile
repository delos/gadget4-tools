GSL_INCL   =  -I$(GSL_HOME)/include
GSL_LIBS   =  -L$(GSL_HOME)/lib  -lgsl -lgslcblas
FFTW_INCL  =  -I$(FFTW_HOME)/include
FFTW_LIBS  =  -L$(FFTW_HOME)/lib  -lfftw3f
HDF5_INCL  =  -I$(HDF5_HOME)/include
HDF5_LIBS  =  -L$(HDF5_HOME)/lib

all: power_of_shell

power_of_shell: power_of_shell.c
	g++ $(FFTW_INCL) $(FFTW_LIBS) -o power_of_shell power_of_shell.c

clean:
  -rm -f power_of_shell
