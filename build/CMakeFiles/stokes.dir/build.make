# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake

# The command to remove a file.
RM = /u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luigi/NMPDE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luigi/NMPDE/build

# Include any dependencies generated for this target.
include CMakeFiles/stokes.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/stokes.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/stokes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stokes.dir/flags.make

CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o: CMakeFiles/stokes.dir/flags.make
CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o: ../src_stokes/stokes_main.cpp
CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o: CMakeFiles/stokes.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luigi/NMPDE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o -MF CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o.d -o CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o -c /home/luigi/NMPDE/src_stokes/stokes_main.cpp

CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.i"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luigi/NMPDE/src_stokes/stokes_main.cpp > CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.i

CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.s"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luigi/NMPDE/src_stokes/stokes_main.cpp -o CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.s

CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o: CMakeFiles/stokes.dir/flags.make
CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o: ../src_stokes/Stokes.cpp
CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o: CMakeFiles/stokes.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luigi/NMPDE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o -MF CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o.d -o CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o -c /home/luigi/NMPDE/src_stokes/Stokes.cpp

CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.i"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luigi/NMPDE/src_stokes/Stokes.cpp > CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.i

CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.s"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luigi/NMPDE/src_stokes/Stokes.cpp -o CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.s

# Object files for target stokes
stokes_OBJECTS = \
"CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o" \
"CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o"

# External object files for target stokes
stokes_EXTERNAL_OBJECTS =

stokes: CMakeFiles/stokes.dir/src_stokes/stokes_main.cpp.o
stokes: CMakeFiles/stokes.dir/src_stokes/Stokes.cpp.o
stokes: CMakeFiles/stokes.dir/build.make
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/lib/libdeal_II.so.9.3.1
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_iostreams.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_serialization.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_system.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_thread.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_regex.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_chrono.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_date_time.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_atomic.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/librol.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/librythmos.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libmuelu-adapters.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libmuelu-interface.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libmuelu.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/liblocathyra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/liblocaepetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/liblocalapack.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libloca.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libnoxepetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libnoxlapack.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libnox.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libstratimikos.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libstratimikosbelos.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libstratimikosaztecoo.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libstratimikosamesos.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libstratimikosml.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libstratimikosifpack.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libanasazitpetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libModeLaplace.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libanasaziepetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libanasazi.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libbelosxpetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libbelostpetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libbelosepetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libbelos.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libml.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libifpack.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libamesos.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libgaleri-xpetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libgaleri-epetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libaztecoo.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libisorropia.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libxpetra-sup.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libxpetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libthyratpetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libthyraepetraext.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libthyraepetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libthyracore.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtrilinosss.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtpetraext.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtpetrainout.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtpetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libkokkostsqr.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtpetraclassiclinalg.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtpetraclassicnodeapi.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtpetraclassic.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libepetraext.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libtriutils.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libzoltan.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libepetra.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libsacado.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/librtop.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libkokkoskernels.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchoskokkoscomm.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchoskokkoscompat.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchosremainder.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchosnumerics.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchoscomm.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchosparameterlist.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchosparser.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libteuchoscore.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libkokkosalgorithms.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libkokkoscontainers.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib/libkokkoscore.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/tbb/2021.3.0/lib/libtbb.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/blacs/1.1/lib/libblacs.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/blacs/1.1/lib/libblacsF77init.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/base/lib/libhwloc.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/adol-c/2.7.2/lib64/libadolc.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/arpack/3.8.0/lib/libarpack.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/gsl/2.7/lib/libgsl.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/gsl/2.7/lib/libgslcblas.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/petsc/3.15.1/lib/libslepc.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/petsc/3.15.1/lib/libpetsc.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hypre/2.22.0/lib/libHYPRE.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libcmumps.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libdmumps.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libsmumps.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libzmumps.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libmumps_common.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libpord.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scalapack/2.1.0/lib/libscalapack.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libumfpack.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libklu.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libcholmod.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libbtf.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libccolamd.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libcolamd.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libcamd.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libamd.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libsuitesparseconfig.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/fftw/3.3.9/lib/libfftw3_mpi.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/fftw/3.3.9/lib/libfftw3.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/p4est/2.3.2/lib/libp4est.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/p4est/2.3.2/lib/libsc.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/openblas/0.3.15/lib/libopenblas.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/lib/libptesmumps.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/lib/libptscotchparmetis.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/lib/libptscotch.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/lib/libptscotcherr.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/lib/libesmumps.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/lib/libscotch.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/lib/libscotcherr.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/netcdf/4.8.0/lib/libnetcdf.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/lib/libhdf5hl_fortran.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/lib/libhdf5_fortran.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/lib/libhdf5_hl.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/lib/libhdf5.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/metis/5.1.0/lib/libparmetis.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/metis/5.1.0/lib/libmetis.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/base/lib/libz.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/base/lib/libbz2.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/base/lib/libmpi_usempif08.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/base/lib/libmpi_usempi_ignore_tkr.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/base/lib/libmpi_mpifh.so
stokes: /u/sw/toolchains/gcc-glibc/11.2.0/base/lib/libmpi.so
stokes: CMakeFiles/stokes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luigi/NMPDE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable stokes"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stokes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stokes.dir/build: stokes
.PHONY : CMakeFiles/stokes.dir/build

CMakeFiles/stokes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stokes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stokes.dir/clean

CMakeFiles/stokes.dir/depend:
	cd /home/luigi/NMPDE/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luigi/NMPDE /home/luigi/NMPDE /home/luigi/NMPDE/build /home/luigi/NMPDE/build /home/luigi/NMPDE/build/CMakeFiles/stokes.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stokes.dir/depend

