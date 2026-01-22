#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "CudaUtil.h"
#include "CudaVectorAlgebra.h"

namespace cuda_linalg
{

void vectorMulScalar(double k, 
                     CudaVector<double>& a)
{
  thrust::transform(a.begin(), a.end(), 
                    a.begin(), 
                    scalarMul<double>(k));
}

void vectorMulScalar(double k, 
                     const CudaVector<double>& a, 
                           CudaVector<double>& b)
{
  thrust::transform(a.begin(), a.end(), 
                    b.begin(), 
                    scalarMul<double>(k));
}

void vectorAxpy(double alpha, 
                const CudaVector<double>& a, 
                      CudaVector<double>& b)
{
  const int n = static_cast<int>(a.size());

  const double* d_a_ptr = a.data();
        double* d_b_ptr = b.data();

  CHECK_CUBLAS(
    cublasDaxpy(
      a.getCuBlasHandle(), 
      n, 
      &alpha, 
      d_a_ptr, 
      1, // incr x
      d_b_ptr, 
      1)) // incr y
}

void vectorAdd(const double alpha, 
               const double beta,
               const CudaVector<double>& a, 
               const CudaVector<double>& b,
                     CudaVector<double>& c)
{
  assert(a.size() == b.size());
  assert(b.size() == c.size());

  vectorInit(c, 0.0);

  vectorAxpy(alpha, a, c);
  vectorAxpy(beta, b, c);
}

void vectorElementWiseMult(const CudaVector<double>& a,
                           const CudaVector<double>& b,
                                 CudaVector<double>& c)
{
  thrust::transform(a.begin(), a.end(),
                    b.begin(), 
                    c.begin(),
                    thrust::multiplies<double>());
}

void normalizeVector(CudaVector<double>& a)
{
  double norm2 = compute2Norm(a);
  vectorMulScalar(1.0 / norm2, a);
}

void makeAbsoluteVector(const CudaVector<double>& a,
                              CudaVector<double>& b)
{
  thrust::transform(a.begin(), a.end(),
                    b.begin(),
                    absolute<double>() );
}

double compute1Norm(const CudaVector<double>& a)
{
  double sum_abs
    = thrust::transform_reduce(a.begin(), a.end(), 
                               absolute<double>(), 
                               static_cast<double>(0.0),
                               thrust::plus<double>());
  return sum_abs;
}

double compute2Norm(const CudaVector<double>& a)
{
  double sum = 0.0;
  const int n = static_cast<int>(a.size());

  CHECK_CUBLAS(
    cublasDnrm2(
      a.getCuBlasHandle(),
      n,
      a.data(),
      1, // incr x
      &sum) )

  return sum;
}

double computeVectorSum(const CudaVector<double>& a)
{
  const int n = static_cast<int>(a.size());
  double sum = 0.0;

  CHECK_CUBLAS(
    cublasDasum(
      a.getCuBlasHandle(),
      n,
      a.data(),
      1, // incr x
      &sum) )

  return sum;
}

double innerProduct(const CudaVector<double>& a,
                    const CudaVector<double>& b)
{
  const int n = static_cast<int>(a.size());
  double sum = 0.0;

  CHECK_CUBLAS(
      cublasDdot(
        a.getCuBlasHandle(),
        n,
        a.data(),
        1, /* incr x */
        b.data(),
        1, /* incr y */
        &sum));

  return sum;
}

double computeVectorMax(const CudaVector<double>& a)
{
  return *thrust::max_element(thrust::device, a.begin(), a.end());
}

void vectorInit(CudaVector<double>& a, double x)
{
  thrust::fill(a.begin(), a.end(), x);
}

void printVector(const CudaVector<double>& d_vector, 
                 const std::string& title)
{
  size_t len = d_vector.size();
  std::vector<double> h_vector(len);
  thrust::copy(d_vector.begin(), d_vector.end(),
               h_vector.begin());

  printf("Vector %s\n", title.c_str());
  for(const auto& val : h_vector)
    printf("%f ", val);
  printf("\n");
}

}
