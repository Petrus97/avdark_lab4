#include "util.h"

#include <assert.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <stdint-gcc.h>
#include <stdio.h>
#include <string.h>

static void my_memcpy(char *dst, const char *src, size_t len) {
  // the vector size should be a multiple of 16
  // because we use 16 bytes types here
  assert((len & 0xF) == 0);
  for (int i = 0; i < len; i += 16) {
    // since the char data type is not aligned, we load 16 characters
    // in an unaligned way, they will fill the vector
    __m128i v = _mm_loadu_si128((__m128i *)(src + i));
    // we store the result in the destination
    _mm_storeu_si128((__m128i *)(dst + i), v);
  }
}

static __m128 vector_sum(const __m128 v0, const __m128 v1, const __m128 v2,
                         const __m128 v3) {
  // horizontal add operation
  return _mm_add_ps(_mm_add_ps(v0, v1), _mm_add_ps(v2, v3));
}

void test_good_memcpy(void) {
  char src[16 * 4096];
  char dst[16 * 4096];
  uint64_t i = 0;
  uint8_t letter = 0;
  while (i < 16 * 4096) {
    src[i++] = letter++;
  }

  my_memcpy(dst, src, (16 * 4096));

  assert(memcmp(src, dst, 16 * 4096) == 0);
}

void test_bad_memcpy(void) {
  char src[16 * 4096];
  char dst[16 * 4096];
  uint64_t i = 0;
  uint8_t letter = 0;
  while (i < 16 * 4096) {
    src[i++] = letter++;
  }

  my_memcpy(dst, src, (16 * 4096) - 16);

  assert(memcmp(src, dst, 16 * 4096) != 0);
}

void test_vector_sum(void) {
  // create 4 identical vectors
  float a[4];
  float b[4];
  float c[4];
  float d[4];
  for (size_t i = 0; i < 4; i++) {
    a[i] = b[i] = c[i] = d[i] = i + 0.5;
  }
  __m128 v0 = _mm_load_ps((float *)a);
  __m128 v1 = _mm_load_ps((float *)b);
  __m128 v2 = _mm_load_ps((float *)c);
  __m128 v3 = _mm_load_ps((float *)d);
  __m128 result = vector_sum(v0, v1, v2, v3);

  print_vector_ps(result);
}

void test_vector_multiplication(void) {
  float a[4];
  float b[4];
  float c[4];
  float d[4];
  for (size_t i = 0; i < 4; i++) {
    a[i] = b[i] = c[i] = d[i] = i + 0.5;
  }
  __m128 v0 = _mm_load_ps((float *)a);
  __m128 v1 = _mm_load_ps((float *)b);
  __m128 res = _mm_mul_ps(v0, v1);
  print_vector_ps(v0);
  print_vector_ps(res);
}

void check_suggested_functions(void) {
  float a[4] = {1.0, 2.0, 3.0, 4.0};
  __m128 v = _mm_load_ps((float *)a);

  __m128 res = _mm_hadd_ps(v, v);
  printf("_mm_hadd_ps\n");
  print_vector_ps(res);

  res = _mm_add_ps(v, v);
  printf("_mm_add_ps\n");
  print_vector_ps(res);

  float first = _mm_cvtss_f32(res);
  printf("_mm_cvtss_f32 \n%f\n", first);

  __m128 zero = _mm_setzero_ps();
  printf("_mm_setzero_ps\n");
  print_vector_ps(zero);

  // _mm_dp_ps
  // _MM_TRANSPOSE4_PS
  float mat[4][4] = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
  __m128 row0 = _mm_load_ps(mat[0]);
  __m128 row1 = _mm_load_ps(mat[1]);
  __m128 row2 = _mm_load_ps(mat[2]);
  __m128 row3 = _mm_load_ps(mat[3]);

  _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
  printf("transpose:\n");
  print_vector_ps(row0);
  print_vector_ps(row1);
  print_vector_ps(row2);
  print_vector_ps(row3);

  printf("\n checking _mm_dp_ps with\n");
  print_vector_ps(v);
  printf("\n");

  res = _mm_dp_ps(v, v, 0xF0);
  printf("0x%02x\n", 0xF0);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xF1);
  printf("0x%02x\n", 0xF1);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xF2);
  printf("0x%02x\n", 0xF2);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xF3);
  printf("0x%02x\n", 0xF3);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xF4);
  printf("0x%02x\n", 0xF4);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xF5);
  printf("0x%02x\n", 0xF5);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xF6);
  printf("0x%02x\n", 0xF6);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xF7);
  printf("0x%02x\n", 0xF7);
  print_vector_ps(res);

  res = _mm_dp_ps(v, v, 0xFF);
  printf("0x%02x\n", 0xFF);
  print_vector_ps(res);

  __m128i ret = _mm_set1_epi8(0x20);
  printf("_mm_set1_epi8\n");
  print_vector_epi8(ret);
  printf("_mm_or_si128\n");
  char A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  char B[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  __m128i aa = _mm_loadu_si128((__m128i *)A);
  __m128i bb = _mm_loadu_si128((__m128i *)B);
  __m128i or = _mm_or_si128(aa, bb);
  print_vector_epi8(aa);
  print_vector_epi8(bb);
  print_vector_epi8(or);
}

int main(void) {
  test_bad_memcpy();
  test_good_memcpy();
  // test_vector_sum();
  // check_suggested_functions();
  test_vector_multiplication();
  return 0;
}