/* 
 * integral.cc
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

/* INTEGRAL_H is defined in the command line with
   -DINTEGRAL_H="integral0.h" etc. */
#include INTEGRAL_H

/* the instance of a memory manager */
mem_manager mm;

long long diff_tsc(long long c1, long long c0) {
  /* want to know what the heck is this *(2.7/2.3)?
     => see the exercise page */
  //return (c1 - c0) * (2.7/2.3);
  return (c1 - c0) * (2.7/2.3);
}

#define M0 100
#define N0 100
#define K0 100

int main(int argc, char ** argv) {
  const long max_alloc_per_iter = 128*16;
  mm.init(max_alloc_per_iter);


  // long n = (argc > 1 ? atol(argv[1]) : 100000000);
  mat<M0,N0> x1(M0);
  mat<N0,K0> x2(N0);
  mat<M0,K0> x3(M0);

  for (long i = 0; i < M0; i++) {
    for (long j = 0; j < N0; j++) {
      x1(i,j) = i - j;
    }
  }
  for (long i = 0; i < N0; i++) {
    for (long j = 0; j < K0; j++) {
      x2(i,j) = i + j;
    }
  }
  for (long i = 0; i < N0; i++) {
    for (long j = 0; j < K0; j++) {
      x3(i,j) = 0;
    }
  }


  long long c0 = _rdtsc();
  x3 = x1 * x2;
  long long c1 = _rdtsc();
  double clks = diff_tsc(c1, c0);
  double NMK = M0 * N0 * K0;
  printf(" %.0f NMK\n", NMK);
  printf("%.0f clocks\n", clks);
  printf("%f clocks/NMK\n", clks/NMK);
  printf("(0, 0): %f\n", x3(0,0));
  printf("(1, 1): %f\n", x3(1,1));
  printf("(99, 0): %f\n", x3(99,0));
  printf("(50, 50): %f\n", x3(50,50));
  printf("(0, 99): %f\n", x3(0,99));
  printf("(99, 99): %f\n", x3(99,99));
  return 0;
}
