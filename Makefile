GSL_INCL   =  -I$(GSL_HOME)/include
GSL_LIBS   =  -L$(GSL_HOME)/lib  -lgsl -lgslcblas
FFTW_INCL  =  -I$(FFTW_HOME)/include
FFTW_LIBS  =  -L$(FFTW_HOME)/lib  -lfftw3f
HDF5_INCL  =  -I$(HDF5_HOME)/include
HDF5_LIBS  =  -L$(HDF5_HOME)/lib

all: powers fields peaks

powers: power power_of_shell_corrected power_of_shell_corrected_frommesh power_of_sphere_frommesh

fields: gen_field potential_field

peaks: find_peaks pos_to_peak peak_parameters peak_profile tidal_tensor_profile tidal_tensor_peaks

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

find_peaks: ./src/find_peaks.c
	g++ -lm -o ./bin/find_peaks ./src/find_peaks.c

pos_to_peak: ./src/pos_to_peak.c
	g++ -lm -o ./bin/pos_to_peak ./src/pos_to_peak.c

peak_parameters: ./src/peak_parameters.c
	mpicc $(FFTW_INCL) $(FFTW_LIBS) -lfftw3f_mpi $(GSL_INCL) $(GSL_LIBS) -lm -o ./bin/peak_parameters ./src/peak_parameters.c

peak_profile: ./src/peak_profile.c
	mpicc $(FFTW_INCL) $(FFTW_LIBS) -lfftw3f_mpi $(GSL_INCL) $(GSL_LIBS) -lm -o ./bin/peak_profile ./src/peak_profile.c

tidal_tensor_profile: ./src/tidal_tensor_profile.c
	mpicc $(FFTW_INCL) $(FFTW_LIBS) -lfftw3f_mpi $(GSL_INCL) $(GSL_LIBS) -lm -o ./bin/tidal_tensor_profile ./src/tidal_tensor_profile.c

tidal_tensor_peaks: ./src/tidal_tensor_peaks.c
	mpicc $(FFTW_INCL) $(FFTW_LIBS) -lfftw3f_mpi $(GSL_INCL) $(GSL_LIBS) -lm -o ./bin/tidal_tensor_peaks ./src/tidal_tensor_peaks.c


