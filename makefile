.PHONY: clean all all_test1 all_test2 install_dependencies tables performance compile0 compile1 compile2 compile3 test0 test1 test2 test3 test00 test01 test02 test03

VALS = 200 2000 20000
THREADS = 1 2 4 8
MPI_PROCESSES = 2 4 8 16
SEED = 123
SEED1 = 345

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
	pip3 install -r requirements.txt

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
	@for numValues in $(VALS); do \
        ./build/opt0/RBSequential0 $${numValues} $(SEED) ./data/SequentialTestResult/opt0/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt0/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt0/RBParallel_CUDA_OpenMP0 $${ompNumThreads} $${numValues} $(SEED) ./data/CUDAOpenMPTestResult/opt0/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt0/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt0/RBParallel_MPI_OpenMP0 $${ompNumThreads} $${numValues} $(SEED) ./data/MPIOpenMPTestResult/opt0/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt0/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

test1:
	@for numValues in $(VALS); do \
        ./build/opt1/RBSequential1 $${numValues} $(SEED) ./data/SequentialTestResult/opt1/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt1/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt1/RBParallel_CUDA_OpenMP1 $${ompNumThreads} $${numValues} $(SEED) ./data/CUDAOpenMPTestResult/opt1/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt1/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt1/RBParallel_MPI_OpenMP1 $${ompNumThreads} $${numValues} $(SEED) ./data/MPIOpenMPTestResult/opt1/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt1/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

test2:
	@for numValues in $(VALS); do \
        ./build/opt2/RBSequential2 $${numValues} $(SEED) ./data/SequentialTestResult/opt2/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt2/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt2/RBParallel_CUDA_OpenMP2 $${ompNumThreads} $${numValues} $(SEED) ./data/CUDAOpenMPTestResult/opt2/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt2/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt2/RBParallel_MPI_OpenMP2 $${ompNumThreads} $${numValues} $(SEED) ./data/MPIOpenMPTestResult/opt2/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt2/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

test3:
	@for numValues in $(VALS); do \
        ./build/opt3/RBSequential3 $${numValues} $(SEED) ./data/SequentialTestResult/opt3/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt3/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt3/RBParallel_CUDA_OpenMP3 $${ompNumThreads} $${numValues} $(SEED) ./data/CUDAOpenMPTestResult/opt3/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt3/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt3/RBParallel_MPI_OpenMP3 $${ompNumThreads} $${numValues} $(SEED) ./data/MPIOpenMPTestResult/opt3/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt3/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

all_test1: test0 test1 test2 test3

test00:
	@for numValues in $(VALS); do \
        ./build/opt0/RBSequential0 $${numValues} $(SEED1) ./data/SequentialTestResult/opt0/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt0/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt0/RBParallel_CUDA_OpenMP0 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt0/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt0/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt0/RBParallel_MPI_OpenMP0 $${ompNumThreads} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt0/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt0/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

test01:
	@for numValues in $(VALS); do \
        ./build/opt1/RBSequential1 $${numValues} $(SEED1) ./data/SequentialTestResult/opt1/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt1/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt1/RBParallel_CUDA_OpenMP1 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt1/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt1/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt1/RBParallel_MPI_OpenMP1 $${ompNumThreads} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt1/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt1/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

test02:
	@for numValues in $(VALS); do \
        ./build/opt2/RBSequential2 $${numValues} $(SEED1) ./data/SequentialTestResult/opt2/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt2/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt2/RBParallel_CUDA_OpenMP2 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt2/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt2/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt2/RBParallel_MPI_OpenMP2 $${ompNumThreads} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt2/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt2/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

test03:
	@for numValues in $(VALS); do \
        ./build/opt3/RBSequential3 $${numValues} $(SEED1) ./data/SequentialTestResult/opt3/test_sequential_$${numValues}.txt ./data/SequentialCSVResult/opt3/csv_sequential_$${numValues}.csv; \
    done
	@for ompNumThreads in $(THREADS); do \
        for numValues in $(VALS); do \
            ./build/opt3/RBParallel_CUDA_OpenMP3 $${ompNumThreads} $${numValues} $(SEED1) ./data/CUDAOpenMPTestResult/opt3/test_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.txt ./data/CUDAOpenMPCSVResult/opt3/csv_CUDA_OpenMP_$${ompNumThreads}_$${numValues}.csv; \
        done \
    done
	@for mpiProc in $(MPI_PROCESSES); do \
        for ompNumThreads in $(THREADS); do \
            for numValues in $(VALS); do \
                mpiexec -np $${mpiProc} ./build/opt3/RBParallel_MPI_OpenMP3 $${ompNumThreads} $${numValues} $(SEED1) ./data/MPIOpenMPTestResult/opt3/test_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.txt ./data/MPIOpenMPCSVResult/opt3/csv_MPI_OpenMP_$${mpiProc}_$${ompNumThreads}_$${numValues}.csv; \
            done \
        done \
    done

all_test2: test00 test01 test02 test03

all_test: test0 test1 test2 test3 test00 test01 test02 test03

tables:
	python ./src/CreateTables.py

performance:
	python ./src/Performance.py