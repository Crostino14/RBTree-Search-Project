.PHONY: clean all all_test1 all_test2 install_dependencies tables performance compile0 compile1 compile2 compile3 test0 test1 test2 test3 test00 test01 test02 test03

.DEFAULT_GOAL := help

VALS1 = 200000 2000000 20000000
VALS2 = 20000 200000 2000000
SEED1 = 123
SEED2 = 345
MAX_THREADS = 12
THREADS = 1 2 4 12
MPI_PROCESSES = 1 2 4

VALS1 = 200000 2000000 20000000
VALS2 = 20000 200000 2000000
SEED1 = 123
SEED2 = 345
MAX_THREADS = 12
THREADS = 1 2 4 12
MPI_PROCESSES = 1 2 4

# === Colors/Decorations for the menu ===
ESC := \033
RESET := $(ESC)[0m
BOLD := $(ESC)[1m
DIM := $(ESC)[2m
FG1 := $(ESC)[38;5;39m
FG2 := $(ESC)[38;5;208m
FG3 := $(ESC)[38;5;70m
FG4 := $(ESC)[38;5;244m
HEAD := $(ESC)[48;5;236m$(ESC)[38;5;81m
BOX  := $(ESC)[38;5;240m

help:
	@printf "$(HEAD)                                                                                        $(RESET)\n"
	@printf "$(HEAD)  RBTree-Search-Project — Make Commands Menu                                           $(RESET)\n"
	@printf "$(HEAD)                                                                                        $(RESET)\n"
	@printf "$(BOX)────────────────────────────────────────────────────────────────────────────────────────$(RESET)\n"
	@printf "$(BOLD)Usage:$(RESET)  make $(FG1)<target>$(RESET)\n\n"
	@printf "$(BOLD)Main targets:$(RESET)\n"
	@awk 'BEGIN {FS":.*## "}; /^[a-zA-Z0-9_\/\.\-]+:.*## / { printf "  $(FG1)%-22s$(RESET)  %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\n$(BOLD)Useful variables:$(RESET)\n"
	@printf "  $(FG3)VALS1$(RESET)          = $(VALS1)\n"
	@printf "  $(FG3)VALS2$(RESET)          = $(VALS2)\n"
	@printf "  $(FG3)SEED1$(RESET)          = $(SEED1)\n"
	@printf "  $(FG3)SEED2$(RESET)          = $(SEED2)\n"
	@printf "  $(FG3)THREADS$(RESET)        = $(THREADS)\n"
	@printf "  $(FG3)MPI_PROCESSES$(RESET)  = $(MPI_PROCESSES)\n"
	@printf "  $(FG3)MAX_THREADS$(RESET)    = $(MAX_THREADS)\n"
	@printf "\n$(BOLD)Examples:$(RESET)\n"
	@printf "  make all            $(FG4)# build all optimization levels O0..O3$(RESET)\n"
	@printf "  make test3          $(FG4)# run tests for O3 binaries$(RESET)\n"
	@printf "  make tables         $(FG4)# generate result tables$(RESET)\n"
	@printf "  make performance    $(FG4)# create performance plots$(RESET)\n"
	@printf "$(BOX)────────────────────────────────────────────────────────────────────────────────────────$(RESET)\n"

clean:
	rm -rf ./build/opt0/*
	rm -rf ./build/opt1/*
	rm -rf ./build/opt2/*
	rm -rf ./build/opt3/*
	rm -rf ./data/CUDAOpenMPTestResult/opt0/*
	rm -rf ./data/CUDAOpenMPTestResult/opt1/*
	rm -rf ./data/CUDAOpenMPTestResult/opt2/*
	rm -rf ./data/CUDAOpenMPTestResult/opt3/*
	rm -rf ./data/MPIOpenMPTestResult/opt0/*
	rm -rf ./data/MPIOpenMPTestResult/opt1/*
	rm -rf ./data/MPIOpenMPTestResult/opt2/*
	rm -rf ./data/MPIOpenMPTestResult/opt3/*
	rm -rf ./data/SequentialTestResult/opt0/*
	rm -rf ./data/SequentialTestResult/opt1/*
	rm -rf ./data/SequentialTestResult/opt2/*
	rm -rf ./data/SequentialTestResult/opt3/*
	rm -rf ./data/CUDAOpenMPCSVResult/opt0/*
	rm -rf ./data/CUDAOpenMPCSVResult/opt1/*
	rm -rf ./data/CUDAOpenMPCSVResult/opt2/*
	rm -rf ./data/CUDAOpenMPCSVResult/opt3/*
	rm -rf ./data/MPIOpenMPCSVResult/opt0/*
	rm -rf ./data/MPIOpenMPCSVResult/opt1/*
	rm -rf ./data/MPIOpenMPCSVResult/opt2/*
	rm -rf ./data/MPIOpenMPCSVResult/opt3/*
	rm -rf ./data/SequentialCSVResult/opt0/*
	rm -rf ./data/SequentialCSVResult/opt1/*
	rm -rf ./data/SequentialCSVResult/opt2/*
	rm -rf ./data/SequentialCSVResult/opt3/*
	rm -rf ./plot_and_tables/PerformanceTable/*
	rm -rf ./plot_and_tables/PerformancePlot/*
	rm -rf ./plot_and_tables/Sequential/*
	rm -rf ./plot_and_tables/CUDAOpenMP/*
	rm -rf ./plot_and_tables/MPIOpenMP/*

all: compile0 compile1 compile2 compile3

install_dependencies: 
	pip3 install matplotlib
	pip3 install pandas
	pip3 install numpy

compile0:
	gcc -o ./build/opt0/RBSequential0 ./src/RBSequential.c ./src/RBMatrix.c -O0
	mpicc -o ./build/opt0/RBParallel_MPI_OpenMP0 ./src/RBParallel_MPI_OpenMP.c ./src/RBMatrix.c -fopenmp -O0
	nvcc -o ./build/opt0/RBParallel_CUDA_OpenMP0 ./src/RBParallel_CUDA_OpenMP.cu ./src/RBMatrix.c -Xcompiler -openmp:llvm -O0
compile1:
	gcc -o ./build/opt1/RBSequential1 ./src/RBSequential.c ./src/RBMatrix.c -O1
	mpicc -o ./build/opt1/RBParallel_MPI_OpenMP1 ./src/RBParallel_MPI_OpenMP.c ./src/RBMatrix.c -fopenmp -O1
	nvcc -o ./build/opt1/RBParallel_CUDA_OpenMP1 ./src/RBParallel_CUDA_OpenMP.cu ./src/RBMatrix.c -Xcompiler -openmp:llvm -O1
compile2:
	gcc -o ./build/opt2/RBSequential2 ./src/RBSequential.c ./src/RBMatrix.c -O2
	mpicc -o ./build/opt2/RBParallel_MPI_OpenMP2 ./src/RBParallel_MPI_OpenMP.c ./src/RBMatrix.c -fopenmp -O2
	nvcc -o ./build/opt2/RBParallel_CUDA_OpenMP2 ./src/RBParallel_CUDA_OpenMP.cu ./src/RBMatrix.c -Xcompiler -openmp:llvm -O2
compile3:
	gcc -o ./build/opt3/RBSequential3 ./src/RBSequential.c ./src/RBMatrix.c -O3
	mpicc -o ./build/opt3/RBParallel_MPI_OpenMP3 ./src/RBParallel_MPI_OpenMP.c ./src/RBMatrix.c -fopenmp -O3
	nvcc -o ./build/opt3/RBParallel_CUDA_OpenMP3 ./src/RBParallel_CUDA_OpenMP.cu ./src/RBMatrix.c -Xcompiler -openmp:llvm -O3


test0:
	@for numValues in $(VALS1); do \
        ./build/opt0/RBSequential0 $${numValues} $(SEED1) ./data/SequentialTestResult/opt0/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt0/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS1); do \
            ./build/opt0/RBParallel_CUDA_OpenMP0 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt0/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt0/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompThread in $(THREADS); do \
            if [ $$((mpiProc * ompThread)) -le $(MAX_THREADS) ]; then \
                for numValues in $(VALS1); do \
                    mpiexec -np $${mpiProc} ./build/opt0/RBParallel_MPI_OpenMP0 $${ompThread} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt0/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt0/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
                done \
            fi \
        done \
    done

test1:
	@for numValues in $(VALS1); do \
        ./build/opt1/RBSequential1 $${numValues} $(SEED1) ./data/SequentialTestResult/opt1/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt1/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS1); do \
            ./build/opt1/RBParallel_CUDA_OpenMP1 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt1/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt1/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompThread in $(THREADS); do \
            if [ $$((mpiProc * ompThread)) -le $(MAX_THREADS) ]; then \
                for numValues in $(VALS1); do \
                    mpiexec -np $${mpiProc} ./build/opt1/RBParallel_MPI_OpenMP1 $${ompThread} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt1/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt1/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
                done \
            fi \
        done \
    done

test2:
	@for numValues in $(VALS1); do \
        ./build/opt2/RBSequential2 $${numValues} $(SEED1) ./data/SequentialTestResult/opt2/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt2/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS1); do \
            ./build/opt2/RBParallel_CUDA_OpenMP2 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt2/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt2/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done 
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompThread in $(THREADS); do \
            if [ $$((mpiProc * ompThread)) -le $(MAX_THREADS) ]; then \
                for numValues in $(VALS1); do \
                    mpiexec -np $${mpiProc} ./build/opt2/RBParallel_MPI_OpenMP2 $${ompThread} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt2/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt2/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
                done \
            fi \
        done \
    done

test3:
	@for numValues in $(VALS1); do \
        ./build/opt3/RBSequential3 $${numValues} $(SEED1) ./data/SequentialTestResult/opt3/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt3/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS1); do \
            ./build/opt3/RBParallel_CUDA_OpenMP3 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt3/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt3/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompThread in $(THREADS); do \
            if [ $$((mpiProc * ompThread)) -le $(MAX_THREADS) ]; then \
                for numValues in $(VALS1); do \
                    mpiexec -np $${mpiProc} ./build/opt3/RBParallel_MPI_OpenMP3 $${ompThread} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt3/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt3/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
                done \
            fi \
        done \
    done

all_test1: test0 test1 test2 test3

test00:
	@for numValues in $(VALS2); do \
        ./build/opt0/RBSequential0 $${numValues} $(SEED2) ./data/SequentialTestResult/opt0/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt0/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS2); do \
            ./build/opt0/RBParallel_CUDA_OpenMP0 $${ompNumThreads} $${numValues} $(SEED2) ./data/CUDAOpenMPTestResult/opt0/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt0/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompThread in $(THREADS); do \
            if [ $$((mpiProc * ompThread)) -le $(MAX_THREADS) ]; then \
                for numValues in $(VALS2); do \
                    mpiexec -np $${mpiProc} ./build/opt0/RBParallel_MPI_OpenMP0 $${ompThread} $${numValues} $(SEED2) ./data/MPIOpenMPTestResult/opt0/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt0/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
                done \
            fi \
        done \
    done

test01:
	@for numValues in $(VALS2); do \
        ./build/opt1/RBSequential1 $${numValues} $(SEED2) ./data/SequentialTestResult/opt1/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt1/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS2); do \
            ./build/opt1/RBParallel_CUDA_OpenMP1 $${ompNumThreads} $${numValues} $(SEED2) ./data/CUDAOpenMPTestResult/opt1/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt1/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompThread in $(THREADS); do \
            if [ $$((mpiProc * ompThread)) -le $(MAX_THREADS) ]; then \
                for numValues in $(VALS2); do \
                    mpiexec -np $${mpiProc} ./build/opt1/RBParallel_MPI_OpenMP1 $${ompThread} $${numValues} $(SEED2) ./data/MPIOpenMPTestResult/opt1/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt1/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
                done \
            fi \
        done \
    done

test02:
	@for numValues in $(VALS2); do \
        ./build/opt2/RBSequential2 $${numValues} $(SEED2) ./data/SequentialTestResult/opt2/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt2/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS2); do \
            ./build/opt2/RBParallel_CUDA_OpenMP2 $${ompNumThreads} $${numValues} $(SEED2) ./data/CUDAOpenMPTestResult/opt2/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt2/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS2); do \
                mpiexec -np $${mpiProc} ./build/opt2/RBParallel_MPI_OpenMP2 $${ompThread} $${numValues} $(SEED2) ./data/MPIOpenMPTestResult/opt2/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt2/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
            done \
        done \
    done

test03:
	@for numValues in $(VALS2); do \
        ./build/opt3/RBSequential3 $${numValues} $(SEED2) ./data/SequentialTestResult/opt3/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt3/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS2); do \
            ./build/opt3/RBParallel_CUDA_OpenMP3 $${ompNumThreads} $${numValues} $(SEED2) ./data/CUDAOpenMPTestResult/opt3/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt3/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompThread in $(THREADS); do \
            if [ $$((mpiProc * ompThread)) -le $(MAX_THREADS) ]; then \
                for numValues in $(VALS2); do \
                    mpiexec -np $${mpiProc} ./build/opt3/RBParallel_MPI_OpenMP3 $${ompThread} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt3/test_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt3/csv_MPI_OpenMP_$${mpiProc}_$${ompThread}_$${numValues}.csv; \
                done \
            fi \
        done \
    done

all_test2: test00 test01 test02 test03

tables:
	python ./src/CreateTables.py

performance:

	python ./src/Performance.py
