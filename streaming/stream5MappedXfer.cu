/*
 *
 * stream5MappedXfer.cu
 *
 * Formulation of stream1Async.cu that uses mapped pinned memory to 
 * perform a computation on inputs in host memory while writing the 
 * output to device memory.
 *
 * Build with: nvcc -I ../chLib stream5MappedXfer.cu
 *
 * Copyright (c) 2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <chError.h>
#include <chCommandLine.h>
#include <chTimer.h>

#include <stdio.h>
#include <stdlib.h>

#include "saxpyCPU.h"
#include "saxpyGPU.cuh"

cudaError_t
MeasureTimes( 
    float *msTotalGPU,
    float *msTotalWallClock,
    size_t N, 
    float alpha,
    int nBlocks, 
    int nThreads )
{
    cudaError_t status;
    chTimerTimestamp chStart, chStop;
    float *dptrOut = 0, *hptrOut = 0;
    float *dptrY = 0, *hptrY = 0;
    float *dptrX = 0, *hptrX = 0;
    cudaEvent_t evStart = 0;
    cudaEvent_t evStop = 0;

    CUDART_CHECK( cudaHostAlloc( &hptrOut, N*sizeof(float), cudaHostAllocMapped ) );
    CUDART_CHECK( cudaMalloc( &dptrOut, N*sizeof(float) ) );
    CUDART_CHECK( cudaMemset( dptrOut, 0, N*sizeof(float) ) );
//    CUDART_CHECK( cudaHostGetDevicePointer( &dptrOut, hptrOut, 0 ) );
    CUDART_CHECK( cudaMemset( dptrOut, 0, N*sizeof(float) ) );
    CUDART_CHECK( cudaHostAlloc( &hptrY, N*sizeof(float), cudaHostAllocMapped ) );
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrY, hptrY, 0 ) );
    CUDART_CHECK( cudaHostAlloc( &hptrX, N*sizeof(float), cudaHostAllocMapped ) );
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrX, hptrX, 0 ) );

    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );
    for ( size_t i = 0; i < N; i++ ) {
        hptrX[i] = (float) rand() / RAND_MAX;
        hptrY[i] = (float) rand() / RAND_MAX;
    }
    

    //
    // begin timing
    //
    chTimerGetTime( &chStart );

    CUDART_CHECK( cudaEventRecord( evStart, 0 ) );
        saxpyGPU<<<nBlocks, nThreads>>>( dptrOut, dptrX, dptrY, N, alpha );
    CUDART_CHECK( cudaEventRecord( evStop, 0 ) );
    CUDART_CHECK( cudaMemcpy( hptrOut, dptrOut, N*sizeof(float), cudaMemcpyDeviceToHost ) );

    //
    // end timing
    //

    chTimerGetTime( &chStop );
    *msTotalWallClock = 1000.0f*chTimerElapsedTime( &chStart, &chStop );

    for ( size_t i = 0; i < N; i++ ) {
        if ( fabsf( hptrOut[i] - (alpha*hptrX[i]+hptrY[i]) ) > 1e-5f ) {
            status = cudaErrorUnknown;
            goto Error;
        }
    }
    CUDART_CHECK( cudaEventElapsedTime( msTotalGPU, evStart, evStop ) );
Error:
    cudaEventDestroy( evStop );
    cudaEventDestroy( evStart );
    cudaFree( dptrOut );
    cudaFreeHost( hptrOut );
    cudaFreeHost( hptrX );
    cudaFreeHost( hptrY );
    return status;
}

double
Bandwidth( float ms, double NumBytes )
{
    return NumBytes / (1000.0*ms);
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int N_Mfloats = 128;
    size_t N;
    int nBlocks = 1500;
    int nThreads = 256;
    float alpha = 2.0f;

    chCommandLineGet( &nBlocks, "nBlocks", argc, argv );
    chCommandLineGet( &nThreads, "nThreads", argc, argv );
    chCommandLineGet( &N_Mfloats, "N", argc, argv );
    printf( "Measuring times with %dM floats", N_Mfloats );
    if ( N_Mfloats==128 ) {
        printf( " (use --N to specify number of Mfloats)");
    }
    printf( "\n" );

    N = 1048576*N_Mfloats;

    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    {
        float msTotalGPU, msWallClock;
        CUDART_CHECK( MeasureTimes( &msTotalGPU, &msWallClock, N, alpha, nBlocks, nThreads ) );
        printf( "Total time (GPU event):  %.2f ms (%.2f MB/s)\n", msTotalGPU, Bandwidth( msTotalGPU, 3*N*sizeof(float) ) );
        printf( "Total time (wall clock): %.2f ms (%.2f MB/s)\n", msWallClock, Bandwidth( msWallClock, 3*N*sizeof(float) ) );
    }

Error:
    if ( status == cudaErrorMemoryAllocation ) {
        printf( "Memory allocation failed\n" );
    }
    else if ( cudaSuccess != status ) {
        printf( "Failed\n" );
    }
    return cudaSuccess != status;
}
