# Makefile for NCCL flow info example

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
BUILDDIR ?= $(abspath ../build)

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc

CXXFLAGS  := -I$(BUILDDIR)/include -I../src/include -I../src -fPIC -std=c++11
LDFLAGS   := -L$(BUILDDIR)/lib -L$(CUDA_LIB) -lnccl -lcudart -lpthread -lrt -ldl

EXAMPLES := flow_info_example

.PHONY: all clean

all: $(EXAMPLES)

flow_info_example: flow_info_example.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(EXAMPLES)