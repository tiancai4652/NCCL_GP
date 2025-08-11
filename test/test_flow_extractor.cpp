/*************************************************************************
 * Copyright (c) 2025, NCCL-SHARP Project. All rights reserved.
 *
 * Test program for NCCL Flow Extractor
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "flow_extractor.h"

// 使用NCCL内置的宏，但需要避免重定义警告
#undef CUDACHECK
#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#ifndef TEST_NCCLCHECK
#undef NCCLCHECK
#define TEST_NCCLCHECK(cmd) do {                    \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

int main(int argc, char* argv[]) {
    setenv("NCCL_DEBUG", "WARN", 1);
    printf("NCCL Flow Extractor Test Suite\n");
    printf("===============================\n");

    int nDev = 3;
    char* env = getenv("GPU_DEV_NUM");
    if (env) {
        nDev = atoi(env);
        printf("Using %d devices (from GPU_DEV_NUM)\n", nDev);
    } else {
        printf("Using %d devices (default)\n", nDev);
    }

    ncclComm_t* comms = (ncclComm_t*)malloc(nDev * sizeof(ncclComm_t));
    int* devs = (int*)malloc(nDev * sizeof(int));
    for (int i = 0; i < nDev; i++) devs[i] = i;

    printf("Initializing NCCL communicators...\n");
    TEST_NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    printf("NCCL initialization complete\n");

    ncclComm_t comm = comms[0];

    TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(1));
    const size_t count = 1024;
    TEST_NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        TEST_NCCLCHECK(ncclAllReduce((const void*)NULL, (void*)NULL, count, ncclFloat, ncclSum, comms[i], (cudaStream_t)0));
    }
    TEST_NCCLCHECK(ncclGroupEnd());
    printf("Real NCCL call done. Records are written under output/<topo>/...\n");

    TEST_NCCLCHECK(ncclWriteAggregatedFlow(comm));

    for (int i = 0; i < nDev; i++) if (comms[i]) ncclCommDestroy(comms[i]);
    free(comms);
    free(devs);

    printf("Done.\n");
    return 0;
} 