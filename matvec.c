/**
 * SSE matrix vector multiplication.
 *
 *
 * Course: Advanced Computer Architecture, Uppsala University
 * Course Part: Lab assignment 4
 *
 * Author: Andreas Sandberg <andreas.sandberg@it.uu.se>
 *
 * $Id: matmul.c 601 2010-08-02 15:06:56Z ansan501 $
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <smmintrin.h>

#ifndef __SSE4_1__
#error This example requires SSE4.1
#endif

#include "util.h"

/* Size of the matrices to multiply */
#define SIZE2 14
#define SIZE (1 << SIZE2)

#define MINDEX(n, m) (((n) << SIZE2) | (m))

#define XMM_ALIGNMENT_BYTES 16 

static float *mat_a __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *vec_b __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *vec_c __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *vec_ref __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

static void
matvec_sse()
{
        /* Assume that the data size is an even multiple of the 128 bit
         * SSE vectors (i.e. 4 floats) */
        assert(!(SIZE & 0x3));

        /* TASK: Implement your SSE version of the matrix-vector
         * multiplication here. (Multiply mat_a and vec_b into vec_c.)
         */
        /* HINT: You might find at least the following instructions
         * useful:
         *  - _mm_setzero_ps
         *  - _mm_load_ps
         *  - _mm_hadd_ps
         *  - _mm_cvtss_f32
         *
         * HINT: You can create the sum of all elements in a vector
         * using two hadd instructions.
         */
        for (int i = 0; i < SIZE; i++)
        {
                for (int j = 0; j < SIZE; j+=4)
                {
                        // // unrolled version
                        // vec_c[i] += (mat_a[MINDEX(i, j + 0)] * vec_b[j + 0])
                        //         + (mat_a[MINDEX(i, j + 1)] * vec_b[j + 1])
                        //         + (mat_a[MINDEX(i, j + 2)] * vec_b[j + 2])
                        //         + (mat_a[MINDEX(i, j + 3)] * vec_b[j + 3]);
                        /**
                         * Load the 4 float elements and multiply them element per element
                         * i.e A_row = {5,2,1,4} b = {3, 1, 0, 2}
                         * result = {15, 2, 0, 8}
                         */
                        __m128 mult_vec = _mm_mul_ps(_mm_load_ps(&mat_a[MINDEX(i,j)]), _mm_load_ps(&vec_b[j]));
                        /**
                         * Do an horizontal sum, so we have
                         * {17, 8, 17, 8}
                         */
                        __m128 first_add = _mm_hadd_ps(mult_vec, mult_vec);
                        /**
                         * Do another horizontal sum on the result of before (to add four numbers)
                         * {23, 23, 23, 23}
                         * and take the first one to store in vec_c
                         */
                        vec_c[i] += _mm_cvtss_f32(_mm_hadd_ps(first_add, first_add));
                }
                
        }
        
}

/**
 * Reference implementation of the matrix vector multiply
 * algorithm. Used to verify the answer. Do NOT change this function.
 */
static void
matvec_ref()
{
        int i, j;

        for (i = 0; i < SIZE; i++)
                for (j = 0; j < SIZE; j++)
                        vec_ref[i] += mat_a[MINDEX(i, j)] * vec_b[j];
}

/**
 * Function used to verify the result. No need to change this one.
 */
static int
verify_result()
{
        float e_sum;
        int i;

        e_sum = 0;
        for (i = 0; i < SIZE; i++) {
                e_sum += vec_c[i] < vec_ref[i] ?
                        vec_ref[i] - vec_c[i] :
                        vec_c[i] - vec_ref[i];
        }

        printf("e_sum: %.e\n", e_sum);

        return e_sum < 1E-6;
}

/**
 * Initialize mat_a and vec_b with "random" data. Write to every
 * element in mat_c to make sure that the kernel allocates physical
 * memory to every page in the matrix before we start doing
 * benchmarking.
 */
static void
init()
{
        int i, j;

        mat_a = _mm_malloc(sizeof(*mat_a) * SIZE * SIZE, XMM_ALIGNMENT_BYTES);
        vec_b = _mm_malloc(sizeof(*vec_b) * SIZE, XMM_ALIGNMENT_BYTES);
        vec_c = _mm_malloc(sizeof(*vec_c) * SIZE, XMM_ALIGNMENT_BYTES);
        vec_ref = _mm_malloc(sizeof(*vec_ref) * SIZE, XMM_ALIGNMENT_BYTES);

        if (!mat_a || !vec_b || !vec_c || !vec_ref) {
                fprintf(stderr, "Memory allocation failed\n");
                abort();
        }

        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++)
                        mat_a[MINDEX(i, j)] = ((7 * i + j) & 0x0F) * 0x1P-2F;
                vec_b[i] = ((i * 17) & 0x0F) * 0x1P-2F;
        }

        memset(vec_c, 0, sizeof(*vec_c) * SIZE);
        memset(vec_ref, 0, sizeof(*vec_ref) * SIZE);
}

static int
run_multiply()
{
        struct timespec ts_start, ts_stop;
        double runtime_ref, runtime_sse;

        printf("Starting SSE run...\n");
        util_monotonic_time(&ts_start);
        /* vec_c = mat_a * vec_b */
        matvec_sse();
        util_monotonic_time(&ts_stop);
        runtime_sse = util_time_diff(&ts_start, &ts_stop);
        printf("SSE run completed in %.2f s\n",
               runtime_sse);

        printf("Starting reference run...\n");
        util_monotonic_time(&ts_start);
        matvec_ref();
        util_monotonic_time(&ts_stop);
        runtime_ref = util_time_diff(&ts_start, &ts_stop);
        printf("Reference run completed in %.2f s\n",
               runtime_ref);

        printf("Speedup: %.2f\n",
               runtime_ref / runtime_sse);


        if (verify_result()) {
                printf("OK\n");
                return 0;
        } else {
                printf("MISMATCH\n");
                return 1;
        }
}

void small_test()
{
        float A[4][4] __attribute__((aligned (XMM_ALIGNMENT_BYTES))) = {
                {4,3,3,3},
                {3,5,5,2},
                {0,4,9,2},
                {0,1,9,0}
        };
        float b[4] __attribute__((aligned (XMM_ALIGNMENT_BYTES))) = {4,8,5,0};

        float x[4] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

        // simple version
        for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                        x[i] += A[i][j] * b[j];

        for (size_t i = 0; i < 4; i++)
        {
                printf("%f ", x[i]);
        }
        printf("\n");
        memset(x, 0, 4 * sizeof(float));
        
        // vectorized
        for (int i = 0; i < 4; i++)
        {
                for (int j = 0; j < 4; j+=4)
                {
                        __m128 mult_vec = _mm_mul_ps(_mm_load_ps(&A[i][j]), _mm_load_ps(&b[j]));
                        __m128 first_add = _mm_hadd_ps(mult_vec, mult_vec);
                        x[i] += _mm_cvtss_f32(_mm_hadd_ps(first_add, first_add));
                }
        }
        for (size_t i = 0; i < 4; i++)
        {
                printf("%f ", x[i]);
        }
        printf("\n");


}

int
main(int argc, char *argv[])
{
        // small_test();
        /* Initialize the matrices with some "random" data. */
        init();

        int rc = run_multiply();
        if (rc)
        {
                goto cleanup;
        }

cleanup:
        _mm_free(mat_a);
        _mm_free(vec_b);
        _mm_free(vec_c);
        _mm_free(vec_ref);

        return rc;
}


/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * indent-tabs-mode: nil
 * c-file-style: "linux"
 * compile-command: "make -k"
 * End:
 */
