/* 
 * mat.h
 */
#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "mem.h"
#include <mtbb/task_group.h>

/* this file defines matrix classes. 

   it currently implements the most straightforward 
   matrix multiply. you are supposed to work mainly
   on this file (not to say you shouldn't change other
   files).
*/

typedef float real;

/* row-major matrix
   (elements in a row are consecutive in memory) */
template<int M, int N>
  struct mat;

/* column-major matrix 
   (elements in a column are consecutive in memory) */
template<int M, int N>
  struct cmat;

template<int M, int N>
struct mat {
  long m;     /* the actual number of rows */
  real * a;
  mat(real * a_, long m_) {
    m = m_;
    a = a_;
  }
  mat(long m_) {
    m = m_;
    a = (real*)mm.alloc(sizeof(real) * m * N);
  }
  /* transpose (you get a column-major matrix) */
  cmat<N,M> T() {
    assert(m <= M);
    cmat<N,M> b(a, m);
    return b;
  }
  /* a(i,j) */
  real& operator()(long i, long j) {
    assert(i < m);
    assert(j < N);
    return a[i * N + j];
  }
};

template<int M, int N>
struct cmat {
  long n;
  real * a;
  cmat(real * a_, long n_) {
    n = n_;
    a = a_;
  }
  cmat(long n_) {
    n = n_;
    a = (real*)mm.alloc(sizeof(real) * M * n);
  }
  /* transpose (you get a row-major matrix) */
  mat<N,M> T() {
    assert(n <= N);
    mat<N,M> b(a, n);
    return b;
  }
  /* a(i,j) */
  real& operator()(long i, long j) {
    assert(i < M);
    assert(j < N);
    return a[i + j * M];
  }
};


/* matrix + matrix
   mat<3,4> a; mat<3,4> b;
   mat<3,4> c = a + b;
 */
template<int M,int N>
mat<M,N> operator+ (mat<M,N> a, mat<M,N> b) {
  mat<M,N> c(a.m);
  assert(a.m == b.m);
  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      c(i,j) = a(i,j) + b(i,j);
    }
  }
  return c;
}

/* matrix - matrix
   mat<3,4> a; mat<3,4> b;
   mat<3,4> c = a - b;
 */
template<int M,int N>
mat<M,N> operator- (mat<M,N> a, mat<M,N> b) {
  mat<M,N> c(a.m);
  assert(a.m == b.m);
  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      c(i,j) = a(i,j) - b(i,j);
    }
  }
  return c;
}

/* matrix -= matrix
   mat<3,4> a; mat<3,4> b;
   a -= b;
 */
template<int M,int N>
mat<M,N> operator-= (mat<M,N> a, mat<M,N> b) {
  assert(a.m == b.m);
  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      a(i,j) -= b(i,j);
    }
  }
  return a;
}

/* scalar * matrix
   mat<3,4> a; 
   mat<3,4> b = 5.6 * a;
 */
template<int M,int N>
mat<M,N> operator* (real k, mat<M,N> a) {
  mat<M,N> b(a.m);
  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      b(i,j) = k * a(i,j);
    }
  }
  return b;
}

/* matrix * matrix (both are row-major)
   mat<3,4> a; mat<4,5> b;
   mat<3,5> c = a * b;
 */
template<int M0,int N0, int K0>
void gemm(mat <M0,K0> a, mat<K0,N0> b, mat<M0,N0> c, long an0, long an1, long am0, long am1, long bn0, long bn1, long bm0, long bm1)
{
  long K = an1 - an0;
  long M = am1 - am0;
  long N = bn1 - bn0;
  assert(K == bm1 - bm0);

  if(M*N + N*K + M*K < 4096) {
    for (long i = 0; i < M; i+= 4) {
      for (long j = 0; j < N; j+= 16) {
        for (long k = 0; k < K; k ++) {
          for (long di = 0; di < 4; di++) {
            if (i+di >= M) break;
            for (long dj = 0; dj < 16; dj += 8){
              if (j+dj+8 < N) {
                c(i+di+am0, j+dj+0+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+0+bn0);
                c(i+di+am0, j+dj+1+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+1+bn0);
                c(i+di+am0, j+dj+2+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+2+bn0);
                c(i+di+am0, j+dj+3+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+3+bn0);
                c(i+di+am0, j+dj+4+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+4+bn0);
                c(i+di+am0, j+dj+5+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+5+bn0);
                c(i+di+am0, j+dj+6+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+6+bn0);
                c(i+di+am0, j+dj+7+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+7+bn0);
              }else{
                if (j+dj+0<N) c(i+di+am0, j+dj+0+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+0+bn0);
                if (j+dj+1<N) c(i+di+am0, j+dj+1+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+1+bn0);
                if (j+dj+2<N) c(i+di+am0, j+dj+2+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+2+bn0);
                if (j+dj+3<N) c(i+di+am0, j+dj+3+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+3+bn0);
                if (j+dj+4<N) c(i+di+am0, j+dj+4+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+4+bn0);
                if (j+dj+5<N) c(i+di+am0, j+dj+5+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+5+bn0);
                if (j+dj+6<N) c(i+di+am0, j+dj+6+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+6+bn0);
                if (j+dj+7<N) c(i+di+am0, j+dj+7+bn0) += a(i+di+am0, k+an0) * b(k+bm0, j+dj+7+bn0);
              }
            }
          }
        }
      }
    }
    // gemm(a, b, c, an0, an1, am0, am1, bn0, bn1, bm0, bm1, cn0, cm0);

  }else if(M >= N && M >= K){
    // 中点
    long am2 = (am0 + am1)/4/2*4;        
    mtbb::task_group tg;
    tg.run([&] { gemm(a, b, c, an0, an1, am0, am2, bn0, bn1, bm0, bm1); });
    tg.run([&] { gemm(a, b, c, an0, an1, am2, am1, bn0, bn1, bm0, bm1); });
    tg.wait();
  }else if(N >= M && N >= K){
    // 中点
    long bn2 = (bn0 + bn1)/4/2*4;
    mtbb::task_group tg;
    tg.run([&] { gemm(a, b, c, an0, an1, am0, am1, bn0, bn2, bm0, bm1); });
    tg.run([&] { gemm(a, b, c, an0, an1, am0, am1, bn2, bn1, bm0, bm1); });
    tg.wait();
  }else{
    // 中点
    long an2 = (an0 + an1)/4/2*4;
    long bm2 = (bm0 + bm1)/4/2*4;
    mtbb::task_group tg;
    tg.run([&] { gemm(a, b, c, an0, an2, am0, am1, bn0, bn1, bm0, bm2); });
    tg.run([&] { gemm(a, b, c, an2, an1, am0, am1, bn0, bn1, bm2, bm1); });
    tg.wait();
  }
}

template<int M,int N, int K>
mat<M,N> operator* (mat<M,K> a, mat<K,N> b) {
  mat<M,N> c(a.m);
  assert(K == b.m);

  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      c(i,j) = 0;
    }
  }
  gemm(a,b,c,0,K,0,M,0,N,0,K);
  return c;
}


/* row-major matrix * column-major matrix 
   (return row-major matrix)
   mat<3,4> a; cmat<4,5> b;
   mat<3,5> c = a * b;
 */
template<int M,int N, int K>
mat<M,N> operator* (mat<M,K> a, cmat<K,N> b) {
  mat<M,N> c(a.m);
  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      c(i,j) = 0;
      for (long k = 0; k < K; k++) {
        c(i,j) += a(i,k) * b(k,j);
      }
    }
  }
  return c;
}

/* column-major matrix * row-major matrix 
   (return row-major matrix)
   mat<3,4> a; cmat<4,5> b;
   mat<3,5> c = a * b;
 */
template<int M,int N, int K>
mat<M,N> operator* (cmat<M,K> a, mat<K,N> b) {
  mat<M,N> c(M);
  assert(a.n == b.m);
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      c(i,j) = 0;
      for (long k = 0; k < a.n; k++) {
  c(i,j) += a(i,k) * b(k,j);
      }
    }
  }
  return c;
}



