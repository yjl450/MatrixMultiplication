#include "bl_config.h"
#define __ARM_FEATURE_SVE
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

#undef profile

//
// C-based micorkernel
//
void bl_dgemm_ukr( 
        int    k,
        int    m,
        int    n,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data 
    ) {
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )
            { 
                // ldc is used here because a[] and b[] are not packed by the
                // starter code
                // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
                //
                c( i, j, ldc ) += a( i, l, ldc) * b( l, j, ldc );   
            }
        }
    }
}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//

/*
Caches (sum of all):   
  L1d:                 64 KiB (1 instance)
  L1i:                 64 KiB (1 instance)
  L2:                  1 MiB (1 instance)
  L3:                  32 MiB (1 instance)
  Cache line:          64 B
*/

void bl_dgemm_okr( 
        int    k, // KC 64
        int    m, // MR 4
        int    n, // NR 4
        double* restrict a,  // MR x KC (m x k) --> L1
        double* restrict b,  // KC x NR (k x n) --> L2
        double* restrict c,  // MR x NR (m x n) --> register
        unsigned long long ldc,
        aux_t* data 
    ) {
    #ifdef profile
    printf("*%dx%d\n", m,n);
    #endif
    register uint32_t ik = 0;

    register svfloat64_t ax;
    svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x;
    // First implementation: fixed register length 256b, 4 floats.
    // If non-divisible, use naive kernel
    // int vl = svcntd();
    // svbool_t pred = svwhilelt_b64_u64(0, vl);
    svbool_t pred = svptrue_b64();
    if (m == 8) {
        #ifdef profile
        printf("#8xn\n");
        #endif
        // if less than 64
        // printf("k=%d, m=%d, n=%d, ldc=%d\n", k, m, n, ldc);

        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        c1x = svld1_f64(svptrue_b64(), c + 1 * ldc);
        c2x = svld1_f64(svptrue_b64(), c + 2 * ldc);
        c3x = svld1_f64(svptrue_b64(), c + 3 * ldc);
        c4x = svld1_f64(svptrue_b64(), c + 4 * ldc);
        c5x = svld1_f64(svptrue_b64(), c + 5 * ldc);
        c6x = svld1_f64(svptrue_b64(), c + 6 * ldc);
        c7x = svld1_f64(svptrue_b64(), c + 7 * ldc);
        
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            aval = *(a + ik * m + 1);
            ax   = svdup_f64(aval);
            c1x  = svmla_f64_m(pred, c1x, ax, bx);

            aval = *(a + ik * m + 2);
            ax   = svdup_f64(aval);
            c2x  = svmla_f64_m(pred, c2x, ax, bx);

            aval = *(a + ik * m + 3);
            ax   = svdup_f64(aval);
            c3x  = svmla_f64_m(pred, c3x, ax, bx);

            aval = *(a + ik * m + 4);
            ax   = svdup_f64(aval);
            c4x  = svmla_f64_m(pred, c4x, ax, bx);

            aval = *(a + ik * m + 5);
            ax   = svdup_f64(aval);
            c5x  = svmla_f64_m(pred, c5x, ax, bx);

            aval = *(a + ik * m + 6);
            ax   = svdup_f64(aval);
            c6x  = svmla_f64_m(pred, c6x, ax, bx);

            aval = *(a + ik * m + 7);
            ax   = svdup_f64(aval);
            c7x  = svmla_f64_m(pred, c7x, ax, bx);

            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);
        svst1_f64(pred, c + 1 * ldc, c1x);
        svst1_f64(pred, c + 2 * ldc, c2x);
        svst1_f64(pred, c + 3 * ldc, c3x);
        svst1_f64(pred, c + 4 * ldc, c4x);
        svst1_f64(pred, c + 5 * ldc, c5x);
        svst1_f64(pred, c + 6 * ldc, c6x);
        svst1_f64(pred, c + 7 * ldc, c7x);

    } else if (m == 4) {
        #ifdef profile
        printf("#4xn\n");
        #endif
        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        c1x = svld1_f64(svptrue_b64(), c + 1 * ldc);
        c2x = svld1_f64(svptrue_b64(), c + 2 * ldc);
        c3x = svld1_f64(svptrue_b64(), c + 3 * ldc);
        // pred = svwhilelt_b64_u64(n, vl); 
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            aval = *(a + ik * m + 1);
            ax   = svdup_f64(aval);
            c1x  = svmla_f64_m(pred, c1x, ax, bx);

            aval = *(a + ik * m + 2);
            ax   = svdup_f64(aval);
            c2x  = svmla_f64_m(pred, c2x, ax, bx);

            aval = *(a + ik * m + 3);
            ax   = svdup_f64(aval);
            c3x  = svmla_f64_m(pred, c3x, ax, bx);

            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);
        svst1_f64(pred, c + 1 * ldc, c1x);
        svst1_f64(pred, c + 2 * ldc, c2x);
        svst1_f64(pred, c + 3 * ldc, c3x);

    } else if (m == 7) {
        #ifdef profile
        printf("#7xn\n");
        #endif
        // if less than 64
        // printf("k=%d, m=%d, n=%d, ldc=%d\n", k, m, n, ldc);

        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        c1x = svld1_f64(svptrue_b64(), c + 1 * ldc);
        c2x = svld1_f64(svptrue_b64(), c + 2 * ldc);
        c3x = svld1_f64(svptrue_b64(), c + 3 * ldc);
        c4x = svld1_f64(svptrue_b64(), c + 4 * ldc);
        c5x = svld1_f64(svptrue_b64(), c + 5 * ldc);
        c6x = svld1_f64(svptrue_b64(), c + 6 * ldc);
        
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            aval = *(a + ik * m + 1);
            ax   = svdup_f64(aval);
            c1x  = svmla_f64_m(pred, c1x, ax, bx);

            aval = *(a + ik * m + 2);
            ax   = svdup_f64(aval);
            c2x  = svmla_f64_m(pred, c2x, ax, bx);

            aval = *(a + ik * m + 3);
            ax   = svdup_f64(aval);
            c3x  = svmla_f64_m(pred, c3x, ax, bx);

            aval = *(a + ik * m + 4);
            ax   = svdup_f64(aval);
            c4x  = svmla_f64_m(pred, c4x, ax, bx);

            aval = *(a + ik * m + 5);
            ax   = svdup_f64(aval);
            c5x  = svmla_f64_m(pred, c5x, ax, bx);

            aval = *(a + ik * m + 6);
            ax   = svdup_f64(aval);
            c6x  = svmla_f64_m(pred, c6x, ax, bx);

            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);
        svst1_f64(pred, c + 1 * ldc, c1x);
        svst1_f64(pred, c + 2 * ldc, c2x);
        svst1_f64(pred, c + 3 * ldc, c3x);
        svst1_f64(pred, c + 4 * ldc, c4x);
        svst1_f64(pred, c + 5 * ldc, c5x);
        svst1_f64(pred, c + 6 * ldc, c6x);

    } else if (m == 6) {
        #ifdef profile
        printf("#6xn\n");
        #endif
        // if less than 64
        // printf("k=%d, m=%d, n=%d, ldc=%d\n", k, m, n, ldc);

        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        c1x = svld1_f64(svptrue_b64(), c + 1 * ldc);
        c2x = svld1_f64(svptrue_b64(), c + 2 * ldc);
        c3x = svld1_f64(svptrue_b64(), c + 3 * ldc);
        c4x = svld1_f64(svptrue_b64(), c + 4 * ldc);
        c5x = svld1_f64(svptrue_b64(), c + 5 * ldc);
        
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            aval = *(a + ik * m + 1);
            ax   = svdup_f64(aval);
            c1x  = svmla_f64_m(pred, c1x, ax, bx);

            aval = *(a + ik * m + 2);
            ax   = svdup_f64(aval);
            c2x  = svmla_f64_m(pred, c2x, ax, bx);

            aval = *(a + ik * m + 3);
            ax   = svdup_f64(aval);
            c3x  = svmla_f64_m(pred, c3x, ax, bx);

            aval = *(a + ik * m + 4);
            ax   = svdup_f64(aval);
            c4x  = svmla_f64_m(pred, c4x, ax, bx);

            aval = *(a + ik * m + 5);
            ax   = svdup_f64(aval);
            c5x  = svmla_f64_m(pred, c5x, ax, bx);

            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);
        svst1_f64(pred, c + 1 * ldc, c1x);
        svst1_f64(pred, c + 2 * ldc, c2x);
        svst1_f64(pred, c + 3 * ldc, c3x);
        svst1_f64(pred, c + 4 * ldc, c4x);
        svst1_f64(pred, c + 5 * ldc, c5x);

    } else if (m == 5) {
        #ifdef profile
        printf("#5xn\n");
        #endif
        // if less than 64
        // printf("k=%d, m=%d, n=%d, ldc=%d\n", k, m, n, ldc);

        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        c1x = svld1_f64(svptrue_b64(), c + 1 * ldc);
        c2x = svld1_f64(svptrue_b64(), c + 2 * ldc);
        c3x = svld1_f64(svptrue_b64(), c + 3 * ldc);
        c4x = svld1_f64(svptrue_b64(), c + 4 * ldc);
        
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            aval = *(a + ik * m + 1);
            ax   = svdup_f64(aval);
            c1x  = svmla_f64_m(pred, c1x, ax, bx);

            aval = *(a + ik * m + 2);
            ax   = svdup_f64(aval);
            c2x  = svmla_f64_m(pred, c2x, ax, bx);

            aval = *(a + ik * m + 3);
            ax   = svdup_f64(aval);
            c3x  = svmla_f64_m(pred, c3x, ax, bx);

            aval = *(a + ik * m + 4);
            ax   = svdup_f64(aval);
            c4x  = svmla_f64_m(pred, c4x, ax, bx);

            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);
        svst1_f64(pred, c + 1 * ldc, c1x);
        svst1_f64(pred, c + 2 * ldc, c2x);
        svst1_f64(pred, c + 3 * ldc, c3x);
        svst1_f64(pred, c + 4 * ldc, c4x);

    } else if (m == 3) {
        #ifdef profile
        printf("#3xn\n");
        #endif
        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        c1x = svld1_f64(svptrue_b64(), c + 1 * ldc);
        c2x = svld1_f64(svptrue_b64(), c + 2 * ldc);
        // pred = svwhilelt_b64_u64(n, vl); 
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            aval = *(a + ik * m + 1);
            ax   = svdup_f64(aval);
            c1x  = svmla_f64_m(pred, c1x, ax, bx);

            aval = *(a + ik * m + 2);
            ax   = svdup_f64(aval);
            c2x  = svmla_f64_m(pred, c2x, ax, bx);
            
            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);
        svst1_f64(pred, c + 1 * ldc, c1x);
        svst1_f64(pred, c + 2 * ldc, c2x);

    } else if (m == 2) {
        #ifdef profile
        printf("#2xn\n");
        #endif
        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        c1x = svld1_f64(svptrue_b64(), c + 1 * ldc);
        // pred = svwhilelt_b64_u64(n, vl); 
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            aval = *(a + ik * m + 1);
            ax   = svdup_f64(aval);
            c1x  = svmla_f64_m(pred, c1x, ax, bx);
            
            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);
        svst1_f64(pred, c + 1 * ldc, c1x);

    } else if (m == 1) {
        #ifdef profile
        printf("#1xn\n");
        #endif
        c0x = svld1_f64(svptrue_b64(), c + 0 * ldc);
        // pred = svwhilelt_b64_u64(n, vl); 
        do {  // 
            register float64_t aval = *(a + ik * m + 0);  // vec_a value
            ax   = svdup_f64(aval);  // vector a is 4 identical aval
            bx   = svld1_f64(svptrue_b64(), b + ik * DGEMM_NR);
            c0x  = svmla_f64_m(pred, c0x, ax, bx);

            ik++;
        } while (ik < k);

        svst1_f64(pred, c + 0 * ldc, c0x);

    } else {        
        #ifdef profile
        printf("#naive\n");
        #endif
        register int im = 0, in = 0;
        for ( ik = 0; ik < k; ik++ ) {  // KC level
            for ( im = 0; im < m; im++ ) {
                for ( in = 0; in < n; in++ ) {  //NR - subpanel B in L2 cache
                  //MR - subpanel A in L1 cache
                    c( im, in, ldc ) += a( ik, im, m ) * b( ik, in, DGEMM_NR );
                    // printf("c( im %d, in %d ) += a( ik %d, im %d, m %d ) * b( ik %d, in %d, n %d )\n", im, in, ldc, ik, im, m, ik, in, n );
                }
            }
        }
    }
}