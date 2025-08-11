#
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#


.PHONY: all clean


TARGETS=src


build:
	g++ -g test_main.cpp -I../build/include  -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -o test_main
	g++ -g test_flow_simple.cpp -I../build/include -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -lm -o test_flow_simple
	g++ -g test_flow_extractor.cpp -I../build/include -I../src -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -o test_flow_extractor
	
clean:
	rm -f test_main test_flow_simple test_flow_extractor *.json 
