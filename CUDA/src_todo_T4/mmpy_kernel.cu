// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else

extern __shared__ _FTYPE_ arr[];

//You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int TW_m = TILESCALE_M * blockDim.x;
    int TW_k = TILESCALE_K * blockDim.x;
    int TW_n = TILESCALE_N * blockDim.x;
    
    _FTYPE_ *As = arr;
    _FTYPE_ *Bs = &arr[TW_m * TW_k];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int I =  by*TW_m + ty;
    int J =  bx*TW_n + tx;

    _FTYPE_ Cij[TILESCALE_M * TILESCALE_N];
    #pragma unroll
    for (unsigned int i =0; i < TILESCALE_M*TILESCALE_N; i++){
        Cij[i] = 0.0f;
    }
    int step = blockDim.x;

    unsigned loop = N/TW_k;
    if (N%TW_k != 0) loop++;
    for(unsigned int kk = 0; kk < loop; kk += 1) {
        for (unsigned int ki =0; ki < TILESCALE_M; ki++){
            for (unsigned int kj =0; kj < TILESCALE_K; kj++){

                if( ((I + step*ki) < N) && (((kk*TW_k) + (tx+step*kj)) < N) ){
                    As[(tx + step * kj) + ((ty + step * ki) * TW_k)] = A[(I + step*ki)*N + (kk*TW_k) + (tx+step*kj)]; 
                } else {
                    As[(tx + step * kj) + ((ty + step * ki) * TW_k)] = 0;
                }

            }
        }

        for (unsigned int ki =0; ki < TILESCALE_K; ki++){
            for (unsigned int kj =0; kj < TILESCALE_N; kj++){    

                if( (((kk*TW_k) + (ty+step*ki)) < N) && ((J + step*kj) < N) ){
                    Bs[(tx + step * kj) + ((ty + step * ki) * TW_n)] = B[((kk*TW_k) + (ty+step*ki)) * N + J + step*kj]; 
                } else {
                    Bs[(tx + step * kj) + ((ty + step * ki) * TW_n)] = 0; 
                }

            }
        }

        __syncthreads();
        for(int k = 0; k < TW_k; k ++){
            for (unsigned int ki =0; ki < TILESCALE_M; ki++){
                for (unsigned int kj =0; kj < TILESCALE_N; kj++){
                    Cij[ki*TILESCALE_M + kj] += As[(ty+step*ki) * TW_k + k] * Bs[k*TW_n + (tx+step *kj)];
                }
            }
        }
    }
    __syncthreads();

    for (unsigned int ki =0; ki < TILESCALE_M; ki++){
        for (unsigned int kj =0; kj < TILESCALE_N; kj++){
            if( ((I+step*ki) < N) && (J + step*kj) < N ) {
                C[(I+step*ki) * N + J + step*kj] = Cij[ki*TILESCALE_M+kj];
            }
        }
    }
}
#endif
