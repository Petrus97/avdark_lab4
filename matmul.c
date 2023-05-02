/**
 * SSE matrix multiplication. Bonus assignment.
 *
 *
 * Course: Advanced Computer Architecture, Uppsala University
 * Course Part: Lab assignment 4
 *
 * Author: Andreas Sandberg <andreas.sandberg@it.uu.se>
 *
 * $Id: matmul.c 70 2011-11-22 10:07:10Z ansan501 $
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <smmintrin.h>

/**
 * Assembly analysis
 * Loop unroll manual: https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AB9U8lJL6yAngGVG6AMKpaAVxYMQADi6kHAGTwGTAA5dwAjTGIJACZSAAdUBUJbBmc3Dz1E5JsBQOCwlkjorjjLTGtUoQImYgJ0908fC0wrXIZq2oJ80Iio2IsauobM3wUh7qDeov7SgEoLVFdiZHYOAFINAEF1mIBmIOQ3LABqdb3HBRYWIIJiIIA6BHPsTZ39w%2BPMM4umBSU6k8Xm8QfssDRgichABJABa2BOMQ0ki8IO2uz24KmJwAGgBZPHGLb%2BaEAcRCeOwIQAKsYAEIATWp2CEJy4ADYzujtuMmDZkCcqLRUHyTiw%2BcYmOsAKx0mHwmUAERlcrhr2lipOpj5dzw4VcBEwpigYjwwGC6BOEHxhOJZIpVNpjOZQjmbvOdLevP5guFovFBGM4RV8vVytloaVWslBF1%2BsNxogpvNmEt1oJRJJ5MpNPpTJZbrmHq9NR9QpFBDFEuQIbVStrCo10Z193jRuMJtoZotVptmftOad%2Bdd7r2np5pbwAvL/olxEwVAbYaXUe1sdbBvbne7qd7Gbt2cdeZdheL3K23qnJwAbqg8Og3gGWK5aMZ/pgIEX0QB2cdbE4AYBAG3CceCkCc2jgQA1meOzbEBQEAPQAFQnBeCjuN8BAIKK2HfOgfJMCcyQAF7fHgCgnIYJyYNejBii%2BNjxPQJyoFQJx4WyMReCc4SEG8CFAahQhCAidHWCQlEQHgDyYA8JySL6FYKHMJzIYhAmCVR751BAYCQKGZwxJyGiqHso5/ppgkoSc1JbEIADSIAnNCLDMZgbCCCcACeSzEMReBuSxAb3KoDG0ExXaiO0JyuMkRhQqJVmCahCBRHJVp4oxeDMd5VaBkRhiWgGQagYIqD5cYyAPF%2B/5aWpGnwVppgsKUPFMMYxCoAA7rB9UtW1vFVS4fXNcYrXceFBCjYJ/D%2BdJPyahoHqgT8jhQnWY6gbsnoxHSqnrL%2ByX1UBc1WlBi0nMtW0Xec62RjdO3nIqkgHUdTUnZ9gFnRA2iXdddIQWtG0Klt2hPXtb1/l9MMnSVNayng9ayuDTZPXsmolVKiPI3St17VdUaoSVwayvjgPLRqKqo8qH2w/TDOATtlXY3SSNU2TRmA1wROVaTeNc2yuM08djNi19zNYyq7PhgLzMxLzJMquTiLC/WdPi5r9WSxKrMy8rgt7IrEr8yrRsc3SItjqLWuCTbttIYhJz%2BCKlqcV1vUaw7CGIYhHUe5d40sMYfroMY8QKBAuzslLOMWyrlOKkW1te97gG%2B1CmCVpxaAZAwQch674eR9Hsd42rGrJ4Dvv22nvvhMNtCB9cb5Z8XEBK5zzPm7LNPgZ3csEwrFt93zBvMzzI9Kv3JvjwTifU0qVe197K8O77z4Rc3web%2B3/s9eBDe58vKd1Wn1mIZvlbPVqLc4egYcRx3jH94xy%2Bp%2BfG%2BMdvxj34/kdX1fhFd%2BcEz7n0dvDaWFdNTo01AXZA14CD/GMFQPYMRn7AJmrbQ6tMwEwxwfbAhP5cElj5FeW895Hx8l3vOKgn4QTvTwYBECYEILQVgvbH6eB/orW4XdEGrwtpIz2jtKGa8EI/VuhjQmN1gYPUBjBERkMziMLTj9P6N8AZA34fIiCEN9oqOhuAhCJVaFQKnmjER0iy76zJsbQMptoFYK1kQphn1XHa2/CQrxaItggXFEEehP4jEASfC%2BN8Sh6GnyAmE18tComWS8RwBYtBODSl4J4DgWhSCoE4OtBQflVhGT2DwUgBBNDJIWFBEA0oND6E4JIDJFScmcF4AoEAdTylZOSaQOAsAkBoDcnQKI5BKCDPiMM6IyBgBcBKaQLA14pyYAAGp4EwN1AA8vERgnBSk0AilEdpEBwjNL4swYg3ldm8DObUbyGzwjaAqF00pgzPIEA2QwWglzunzI8oYYA4gfn4HnJUOi7SfmYFUBUTcVzyCCFaM0rs4RiC3OcFgZpuoWBXIWEKJgwAFCrPWVsnZ3BeD8EECIMQ7ApAyEEIoFQ6gfm6F8AYIwIBTBVX0HqdpkAFioHiO0cFABaDZMQThCpYO%2BSQDxSnlEqHYCADgRieF8AEKYhRihZCSCkAQyqtU5FSD0DVswWhtCqBMPVYxWhPPNV0I1fQSiDC6Jap1dR7UzBKAsApyxVgSBSWkppPzckcBOKoLw7IhXskUsAZAApZkyqtLgQgJBilcDmLwLpWg3SkGqbU%2BpHBGmkEydk4NbSOllIqdm1JHAYiBpLa0it3Ts10WIMkOwkggA%3D
 * My SSE version: https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AB9U8lJL6yAngGVG6AMKpaAVxYMQkgOykHAGTwGTAA5dwAjTGIQAGYATlIAB1QFQlsGZzcPL19k1JsBQOCwlkjo%2BItMKwKGIQImYgJM908fSur0uoaCItCIqNiEhXrG5uy24e7ekrLBgEoLVFdiZHYOAFINAEF1gCYYoOQ3LABqdZjHBRYWIIJiIIA6BHPsTZ39w%2BPMM4umBSVGk8Xm8QfssDRgichABJABa2BOuw0kgAHCDtnsYuCgt8ABoAWXxxi2/mhAHEQvjsCEACrGABCAE0adghCcuAA2M4Y7aTGzIE5UWioJgEE4sUXGJjrACs9Jh8NlABFZfK4a8ZUqTqZRXc8OFXARMKYoGI8MBgugThACUSSeTKdS6UyWUI5u7zvS3ny8AKhSKxRKCMZwqqFRqVXLw8rtVKCHqDUaTRAzRbMFabYTiaSKVTaQzmaz3XNPd76vzBcLReLJcgw%2BrlfXFZrY7r7onjcZTbRzZbrbbsw6887C26PTEvTytj6/VXA5LiJgqE2IyuYzr4%2B3DZ3u730/2s/bc06C67i6WpzOTgA3VB4dBvIMsVy0Yz/TAQEsY7yTrYnf8Af%2BtwnHgpAnNoYEANYXjs2yAYBAD0ABUJzTgo7jfAQCDVlh3zoKKTAnKkABe3x4AoJyGCcmDXow4ovjYiT0CcqBUCcuHsrsKInOEhBvPBgEoUIQgIrR1gkBREB4A8mAPCckiVgGChzCcSEIfxAmUe%2BjQQGAkDhmcuxchoqgxOOv4aQJyEnDSWxCAA0iAJzQiwTGYGwggnAAnksxBEXgrnMUG9yqPRtCMT2og1CcripEYUIiZZAkoQgUSyda%2BIMXgTFeTWwaEYYVpBiGIGCKgeXGMgDxfn%2BmmqepcGaaYLBcFxlHGMQqAAO4wXVzWtdx4SVS4vVNcYLVtc%2B4WjQJ/B%2BVJPxahonogT8jhQg2E4gXsXq7PSKnrD%2BSV1YBc3WpBi0nMtW0Xec63RjdO3nEqkgHUdjUnZ9AFnRA2iXdd9LgWtG2Klt2hPXtb2/l9MMnQhCEVXWcp4I2crgy2T0xFqxXSsjqP0rde1XTGKHFaGcqE4Dy2aqq6Mqh9sOM/%2B8NM/%2BO0Vbj9IozTFOGYDXAkxV5ME3z7L43Tx2s19LOs%2BzOOqtzkYi%2BzuyC2TqqU4i4uNgzUvSwjstE/LeM88rRMxGrkrC5rFumxLE6S3rdWO07VkI/4IpWhxnU9brruaUwHXdZd40sMYVboMYiQKBAewcsbXP45r1NKiWDt%2B/78Es0ImBihxaBZAwofh57Ucx3HCfQXbypp4D8Mu5n/5DQXIfXG%2BudlxA6u8%2BzttK3TYHd2bgOq9XmqD1bGuiwLY9KhPwbW6LKe0zXM2NwJDeN1NYrPdqbdTZ3gc%2B2BzcuLXa/r4B2%2Bt2H2HoJH0ddwxg8MefGeX9fu/F3fD8x9vL/hTfrBWql94LFSRonWefMv5t2QNeAg/xjBUBiLsJ%2BgCL5O0OvTEBsMsEuzwd%2BbBZZRS%2BhvHeB82wnwvg6kuT8IJ3o4MAsBUC4EoIwRdj9PA/0VpcLuiDV4W0UZ7R2lDTes0SDnW4TdYGD1AbQWEZDM4DD14/T%2BrvAGQM%2BGyPAhDfaSjoagLAQuJcCttYY2EVjDmpjTZVy1KTSeFMzEqnTowzB3hsGy3cfgrxhD0RbGAhKIIdDvwGP/FQ18746EuKwRwBYtBOAyl4J4DgWhSCoE4OtBQvlViGRiDwUgBBNCxIWJBEAMoND6E4JIJJRS0mcF4AoEAFTCkpNiaQOAsAkBoFcnQKI5BKDdMSL06IyBgBcDyaQLA15fSYAAGp4EwF1AA8okRgnB8k0HClERpEBwi1N4swYgXl1m8AOQ0LySzwjaEwNYE5pBukeQIEshgtBjmtMme5QwwBxDvPwIuaweBaKNPeZgVQNztx3NuFUWpPZwjEHOc4LAtS9QsBOQsIUTBgAKHmYslZazuC8H4IIEQYh2BSBkIIRQKh1DvN0FwfQXyQCmEqvofUjTIALFQIkGowKAC0SzdgnF5Swd8kgHj5MsDcmo9gGBOBcC0PQAQcQzAGPSvIaQBBjE8GqlIGqGDTH6NEelkqAUCC6KMeV2RjVVClZ0EYPRlWGr0JMC1hdnX2oNaUVVCwsnLFWBIOJCSanvPSRwE4qgUQcl5RyBSwBkACnGeK60uBCASMxFwOYvAWlaHdKQUp5TKkcGqaQZJqTQ0NKaQUopub4kcF2MGst9Sq2tNzbRYgqQ7CSCAA%3D
 */

#ifndef __SSE4_1__
#error This example requires SSE4.1
#endif

#include "util.h"

/* Size of the matrices to multiply */
#define SIZE 2048

#define SSE_BLOCK_SIZE 4

#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE 64
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 1024
#endif

/* A mode that controls how the matrix multiplication is optimized may
 * be specified at compile time. The following modes are defined:
 *
 * MODE_SSE - A simple, non-blocked, implementation of the matrix
 * multiplication.
 *
 * MODE_SSE_BLOCKED - A blocked matrix multiplication with implemented
 * using a 4x4 SSE block.
 *
 * MODE_BLOCKED - Blocked matrix mutliplication using ordinary
 * floating point math.
 */
#define MODE_SSE_BLOCKED 1
#define MODE_SSE 2
#define MODE_BLOCKED 3

#ifndef MODE
#define MODE MODE_SSE
#endif


#define XMM_ALIGNMENT_BYTES 16 

static float mat_a[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_b[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_c[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_ref[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

/**
 * Blocked matrix multiplication, SSE block (4x4 matrix). Implement
 * your solution to the bonus assignment here.
 */
static inline void
matmul_sse_block(int i, int j, int k)
{
        /* BONUS TASK: Implement your SSE 4x4 matrix multiplication
         * block here. */
        /* HINT: You might find at least the following instructions
         * useful:
         *  - _mm_dp_ps
         *  - _MM_TRANSPOSE4_PS
         *
         * HINT: The result of _mm_dp_ps is scalar. The third
         * parameter can be used to restrict to which elements the
         * result is stored, all other elements are set to zero.
         */
}

/**
 * Blocked matrix multiplication, SSE block (4x4 matrix) implemented
 * using ordinary floating point math.
 */
static inline void
matmul_block(int i, int j, int k)
{
        mat_c[i][j] += 
                mat_a[i][k] * mat_b[k][j]
                + mat_a[i][k + 1] * mat_b[k + 1][j]
                + mat_a[i][k + 2] * mat_b[k + 2][j]
                + mat_a[i][k + 3] * mat_b[k + 3][j];

        mat_c[i][j + 1] += 
                mat_a[i][k] * mat_b[k][j + 1]
                + mat_a[i][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i][j + 2] += 
                mat_a[i][k] * mat_b[k][j + 2]
                + mat_a[i][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i][j + 3] += 
                mat_a[i][k] * mat_b[k][j + 3]
                + mat_a[i][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i][k + 3] * mat_b[k + 3][j + 3];



        mat_c[i + 1][j] += 
                mat_a[i + 1][k] * mat_b[k][j]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j];

        mat_c[i + 1][j + 1] += 
                mat_a[i + 1][k] * mat_b[k][j + 1]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i + 1][j + 2] += 
                mat_a[i + 1][k] * mat_b[k][j + 2]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i + 1][j + 3] += 
                mat_a[i + 1][k] * mat_b[k][j + 3]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j + 3];



        mat_c[i + 2][j] += 
                mat_a[i + 2][k] * mat_b[k][j]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j];

        mat_c[i + 2][j + 1] += 
                mat_a[i + 2][k] * mat_b[k][j + 1]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i + 2][j + 2] += 
                mat_a[i + 2][k] * mat_b[k][j + 2]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i + 2][j + 3] += 
                mat_a[i + 2][k] * mat_b[k][j + 3]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j + 3];



        mat_c[i + 3][j] += 
                mat_a[i + 3][k] * mat_b[k][j]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j];

        mat_c[i + 3][j + 1] += 
                mat_a[i + 3][k] * mat_b[k][j + 1]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i + 3][j + 2] += 
                mat_a[i + 3][k] * mat_b[k][j + 2]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i + 3][j + 3] += 
                mat_a[i + 3][k] * mat_b[k][j + 3]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j + 3];

/*
 * The code in this function can alternatively be expressed using macros:
 */
/*
#define BLOCK_SUB(n, m, l) ( mat_a[i + n][k + l] * mat_b[k + l][j + m] )

#define BLOCK_CELL(n, m) { \
        mat_c[i + n][j + m] += \
                BLOCK_SUB(n, m, 0) + \
                BLOCK_SUB(n, m, 1) + \
                BLOCK_SUB(n, m, 2) + \
                BLOCK_SUB(n, m, 3) ; \
}

#define BLOCK_ROW(n) { \
        BLOCK_CELL(n, 0); \
        BLOCK_CELL(n, 1); \
        BLOCK_CELL(n, 2); \
        BLOCK_CELL(n, 3); \
}

        BLOCK_ROW(0);
        BLOCK_ROW(1);
        BLOCK_ROW(2);
        BLOCK_ROW(3);
*/

}

#if MODE == MODE_SSE_BLOCKED || MODE == MODE_BLOCKED
/**
 * Blocked matrix multiplication, L1 block.
 */
static inline void
matmul_block_l1(int i, int j, int k)
{
        int ii, jj, kk;

        for (ii = i; ii < i + L1_BLOCK_SIZE; ii += SSE_BLOCK_SIZE)
                for (kk = k; kk < k + L1_BLOCK_SIZE; kk += SSE_BLOCK_SIZE)
                        for (jj = j; jj < j + L1_BLOCK_SIZE; jj += SSE_BLOCK_SIZE) {
#if MODE == MODE_SSE_BLOCKED
                                matmul_sse_block(ii, jj, kk);
#elif MODE == MODE_BLOCKED
                                matmul_block(ii, jj, kk);
#endif
                        }
}

/**
 * Blocked matrix multiplication, L2 block.
 */
static inline void
matmul_block_l2(int i, int j, int k)
{
        int ii, jj, kk;

        for (ii = i; ii < i + L2_BLOCK_SIZE; ii += L1_BLOCK_SIZE)
                for (kk = k; kk < k + L2_BLOCK_SIZE; kk += L1_BLOCK_SIZE)
                        for (jj = j; jj < j + L2_BLOCK_SIZE; jj += L1_BLOCK_SIZE)
                                matmul_block_l1(ii, jj, kk);
}

/**
 * Blocked matrix multiplication, entry function for multiplying two
 * matrices.
 */
static void
matmul_sse()
{
        int i, j, k;

        for (i = 0; i < SIZE; i += L2_BLOCK_SIZE)
                for (k = 0; k < SIZE; k += L2_BLOCK_SIZE)
                        for (j = 0; j < SIZE; j += L2_BLOCK_SIZE)
                                matmul_block_l2(i, j, k);
}

#elif MODE == MODE_SSE

/**
 * Matrix multiplication. This is the procedure you should try to
 * optimize.
 */
static void
matmul_sse()
{
        int i, j, k;

        /* Assume that the data size is an even multiple of the 128 bit
         * SSE vectors (i.e. 4 floats) */
        assert(!(SIZE & 0x3));

        /* TASK: Implement your simple matrix multiplication using SSE
         * here. (Multiply mat_a and mat_b into mat_c.)
         */
        // __m128 a_row, b_col, mult;
        for (i = 0; i < SIZE; i++) {
                for (k = 0; k < SIZE; k+=4) {
                        // Load the row
                        __m128 a_row = _mm_load_ps(&mat_a[i][k + 0]);
                        for (j = 0; j < SIZE; j+=4) {
                                // /*unrolled*/
                                // mat_c[i][j] += mat_a[i][k + 0] * mat_b[k + 0][j]
                                //                 + mat_a[i][k + 1] * mat_b[k + 1][j]
                                //                 + mat_a[i][k + 2] * mat_b[k + 2][j]
                                //                 + mat_a[i][k + 3] * mat_b[k + 3][j];
                                __m128 row0 = _mm_load_ps(&mat_b[k + 0][j]);
                                __m128 row1 = _mm_load_ps(&mat_b[k + 1][j]);
                                __m128 row2 = _mm_load_ps(&mat_b[k + 2][j]);
                                __m128 row3 = _mm_load_ps(&mat_b[k + 3][j]);
                                _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
                                // Set the column
                                // b_col = _mm_set_ps(mat_b[k + 3][j], mat_b[k + 2][j], mat_b[k + 1][j], mat_b[k + 0][j]);
                                
                                // mult = _mm_mul_ps(a_row, b_col);
                                // mult = _mm_hadd_ps(mult, mult);
                                // mult = _mm_hadd_ps(mult, mult);

                                // mat_c[i][j] += _mm_cvtss_f32(mult);

                                mat_c[i][j + 0] += _mm_cvtss_f32(_mm_dp_ps(a_row, row0, 0xF1));
                                mat_c[i][j + 1] += _mm_cvtss_f32(_mm_dp_ps(a_row, row1, 0xF1));
                                mat_c[i][j + 2] += _mm_cvtss_f32(_mm_dp_ps(a_row, row2, 0xF1));
                                mat_c[i][j + 3] += _mm_cvtss_f32(_mm_dp_ps(a_row, row3, 0xF1));
                        }
                }
        }
}

#else

#error Invalid mode

#endif

/**
 * Reference implementation of the matrix multiply algorithm. Used to
 * verify the answer from matmul_opt. Do NOT change this function.
 */
//#pragma GCC optimize ("O0")
#pragma GCC optimize ("no-tree-vectorize")
static void
matmul_ref()
{
        int i, j, k;

        for (i = 0; i < SIZE; i++) {
                for (k = 0; k < SIZE; k++) {
                        for (j = 0; j < SIZE; j++) {
                                mat_ref[i][j] += mat_a[i][k] * mat_b[k][j];
                        }
                }
        }
}
#pragma GCC optimize ("tree-vectorize")
// #pragma GCC optimize ("")

void print_simple(float matrix[4][4])
{
        for (int i = 0; i < 4; i++)
        {
                for (int j = 0; j < 4; j++)
                {
                        printf("%.2f ", matrix[i][j]);
                }
                printf("\n");
        }
        
}


void simple_test(void)
{
        float A[4][4];
        memset(A, 0, 4 * 4 * sizeof(float));
        float B[4][4] = {
                {5,2,6,1},
                {0,6,2,0},
                {3,8,1,4},
                {1,8,5,6},
        };
        float C[4][4] = {
                {7,5,8,0},
                {1,8,2,6},
                {9,4,3,8},
                {5,3,7,9},
        };

        // simple multiplication
        for (int i = 0; i < 4; i++)
        {
                for (int k = 0; k < 4; k++)
                {
                        for (int j = 0; j < 4; j++)
                        {
                                A[i][j] += B[i][k] * C[k][j];
                        }
                }
        }
        print_simple(A);
        memset(A, 0, 4 * 4 * sizeof(float));
        // unrolled multiplication
        for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                        for (int k = 0; k < 4; k+=4) {
                                A[i][j] += B[i][k + 0] * C[k + 0][j]
                                                + B[i][k + 1] * C[k + 1][j]
                                                + B[i][k + 2] * C[k + 2][j]
                                                + B[i][k + 3] * C[k + 3][j];
                        }
                }
        }
        printf("unrolled\n");
        print_simple(A);
        memset(A, 0, 4 * 4 * sizeof(float));
        for (int i = 0; i < 4; i++) {
                for (int k = 0; k < 4; k+=4) {
                        for (int j = 0; j < 4; j++) {
                                A[i][j] += B[i][k + 0] * C[k + 0][j]
                                                + B[i][k + 1] * C[k + 1][j]
                                                + B[i][k + 2] * C[k + 2][j]
                                                + B[i][k + 3] * C[k + 3][j];
                        }
                }
        }
        printf("Unrolled V2\n");
        print_simple(A);

        memset(A, 0, 4 * 4 * sizeof(float));
        __m128 a_row;
        // __m128 b_col;
        // __m128 mult;
        for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j+=4) {
                        for (int k = 0; k < 4; k+=4) {
                                // mat_c[i][j] += mat_a[i][k + 0] * mat_b[k + 0][j]
                                //                 + mat_a[i][k + 1] * mat_b[k + 1][j]
                                //                 + mat_a[i][k + 2] * mat_b[k + 2][j]
                                //                 + mat_a[i][k + 3] * mat_b[k + 3][j];
                                __m128 row0 = _mm_load_ps(&C[k + 0][j]);
                                __m128 row1 = _mm_load_ps(&C[k + 1][j]);
                                __m128 row2 = _mm_load_ps(&C[k + 2][j]);
                                __m128 row3 = _mm_load_ps(&C[k + 3][j]);
                                _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
                                // Load the row
                                a_row = _mm_load_ps(&B[i][k + 0]);
                                // Set the column_mm_load_ps(&mat_a[k][j]); //
                                // b_col = _mm_set_ps(mat_b[k + 3][j], mat_b[k + 2][j], mat_b[k + 1][j], mat_b[k + 0][j]);
                                
                                // mult = _mm_mul_ps(a_row, row0);
                                // mult = _mm_hadd_ps(mult, mult);
                                // mult = _mm_hadd_ps(mult, mult);
                                // print_vector_ps(row0);
                                // mult = _mm_dp_ps(a_row, row0, 0xF0);

                                A[i][j + 0] += _mm_cvtss_f32(_mm_dp_ps(a_row, row0, 0xF1)); //_mm_cvtss_f32(mult);
                                A[i][j + 1] += _mm_cvtss_f32(_mm_dp_ps(a_row, row1, 0xF1));
                                A[i][j + 2] += _mm_cvtss_f32(_mm_dp_ps(a_row, row2, 0xF1));
                                A[i][j + 3] += _mm_cvtss_f32(_mm_dp_ps(a_row, row3, 0xF1));
                                // print_simple(A);
                        }
                }
        }
        printf("SSE\n");
        print_simple(A);
}

/**
 * Function used to verify the result. No need to change this one.
 */
static int
verify_result()
{
        float e_sum;
        int i, j;

        e_sum = 0;
        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++) {
                        e_sum += mat_c[i][j] < mat_ref[i][j] ?
                                mat_ref[i][j] - mat_c[i][j] :
                                mat_c[i][j] - mat_ref[i][j];
                }
        }

        printf("e_sum: %.e\n", e_sum);

        return e_sum < 1E-6;
}

/**
 * Initialize mat_a and mat_b with "random" data. Write to every
 * element in mat_c to make sure that the kernel allocates physical
 * memory to every page in the matrix before we start doing
 * benchmarking.
 */
static void
init_matrices()
{
        int i, j;

        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++) {
                        mat_a[i][j] = ((i + j) & 0x0F) * 0x1P-4F;
                        mat_b[i][j] = (((i << 1) + (j >> 1)) & 0x0F) * 0x1P-4F;
                }
        }

        memset(mat_c, 0, sizeof(mat_c));
        memset(mat_ref, 0, sizeof(mat_ref));
}

static int
run_multiply()
{
        struct timespec ts_start, ts_stop;
        double runtime_ref, runtime_sse;

        printf("Starting optimized run...\n");
        util_monotonic_time(&ts_start);
        /* mat_c = mat_a * mat_b */
        matmul_sse();
        util_monotonic_time(&ts_stop);
        runtime_sse = util_time_diff(&ts_start, &ts_stop);
        printf("Optimized run completed in %.2f s\n",
               runtime_sse);

        printf("Starting reference run...\n");
        util_monotonic_time(&ts_start);
        matmul_ref();
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

int
main(int argc, char *argv[])
{
        // simple_test();
        // return 0;
        /* Initialize the matrices with some "random" data. */
        init_matrices();

        int rc = run_multiply();
        if (rc) return 1;

        return 0;
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
