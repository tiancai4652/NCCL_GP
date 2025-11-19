#
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# mpicxx -g test_mpi_tp_dp_allreduce.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -Wl,--no-as-needed -lnccl -lcuda -o test_mpi_tp_dp_allreduce
# mpicxx -g test_mpi_tp_dp_allreduce.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -Wl,--no-as-needed -lnccl -o test_mpi_tp_dp_allreduce
# See LICENCE.txt for license information
#


.PHONY: all clean


TARGETS=src


build:
	g++ -g test_main.cpp -I../build/include  -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -ldl -o test_main
	g++ -g test_flow_simple.cpp -I../build/include -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -lm -ldl -o test_flow_simple
	g++ -g test_flow_extractor.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -ldl -o test_flow_extractor
	g++ -g test_tp.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -ldl -o test_tp
	mpic++ -o test_dev_process test_dev_process.cpp -I../fake_cuda/include -I../build/include -L../fake_cuda/lib -L../build/lib -lnccl -Wl,-rpath,../fake_cuda/lib:../build/lib
	# mpic++ -o test_dev_process test_dev_process.cpp -I$CUDA_INC -I$NCCL_ROOT_DIR/build/include -L$CUDA_LIB -L$NCCL_ROOT_DIR/build/lib -lnccl -Wl,-rpath,$CUDA_LIB:$NCCL_ROOT_DIR/build/lib
	# mpicxx -g test_mpi_tp_dp_allreduce.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -Wl,--no-as-needed -lnccl -o test_mpi_tp_dp_allreduce
	# mpicxx -g test_mpi_tp_dp_allreduce.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -Wl,--no-as-needed -Wl,--whole-archive -lnccl -Wl,--no-whole-archive -o test_mpi_tp_dp_allreduce
	# mpicxx -g test_mpi_tp_dp_allreduce.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib ../build/obj/graph/fake_cuda.o -lnccl -lpthread -lrt -ldl -o test_mpi_tp_dp_allreduce
	# mpicxx -g test_dev_process.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -Wl,--no-as-needed -lnccl -o test_dev_process
clean:
	rm -f test_main test_flow_simple test_flow_extractor test_tp_dp_allreduce test_mpi_tp_dp_allreduce *.json 
