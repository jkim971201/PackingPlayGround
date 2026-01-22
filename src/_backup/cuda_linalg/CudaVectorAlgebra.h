#ifndef CUDA_VECTOR_ALGEBRA_H
#define CUDA_VECTOR_ALGEBRA_H

#include <string>
#include <string_view>
#include "CudaVector.h"

namespace cuda_linalg
{

template<typename T>
struct scalarMul
{
  T a;

  scalarMul(T _a) : a(_a) {}

  __host__ __device__ 
  T operator() (const T& x) const
  {
    return a * x;
  }
};

template<typename T>
struct absolute
{
  __host__ __device__
  T operator() (const T& x) const
  {
    if(x >= 0)
      return x;
    else
      return x * -1;
  }
};

/* Scalar Multiplication  a = k * a */
void vectorMulScalar(double k, 
                     CudaVector<double>& a);

/* Scalar Multiplication  b = k * a */
void vectorMulScalar(double k, 
                     const CudaVector<double>& a, 
                           CudaVector<double>& b);

/* Vector Axpy b = b + alpha * a  */
void vectorAxpy(const double alpha, 
                const CudaVector<double>& a, 
                      CudaVector<double>& b);

/* Vector Add c = a + b */
void vectorAdd(const CudaVector<double>& a, 
               const CudaVector<double>& b,
                     CudaVector<double>& c);

/* Vector Add c = alpha * a + beta * b  */
void vectorAdd(const double alpha, 
               const double beta,
               const CudaVector<double>& a, 
               const CudaVector<double>& b,
                     CudaVector<double>& c);

/* Element-wise Multiplication c = a .* b */
void vectorElementWiseMult(const CudaVector<double>& a,
                           const CudaVector<double>& b,
                                 CudaVector<double>& c);

/* Compute a = a / ||a||_2 */
void normalizeVector(CudaVector<double>& a);

/* Make Absolute vector b = |a| */
void makeAbsoluteVector(const CudaVector<double>& a,
                              CudaVector<double>& b);

/* Compute ||a||_1 */
double compute1Norm(const CudaVector<double>& a);

/* Compute ||a||_2 */
double compute2Norm(const CudaVector<double>& a);

/* Compute Sum(a) */
double computeVectorSum(const CudaVector<double>& a);

/* Compute a^T * b */
double innerProduct(const CudaVector<double>& a,
                    const CudaVector<double>& b);

/* Find max(a) */
double computeVectorMax(const CudaVector<double>& a);

template<typename T>
inline const T* getRawPointer(const CudaVector<T>& vec)
{
  return vec.data();
}

template<typename T>
inline T* getRawPointer(CudaVector<T>& vec)
{
  return vec.data();
}

void vectorInit(CudaVector<double>& a, double x);

void printVector(const CudaVector<double>& d_vector, 
                 const std::string& title = "");

}

#endif
