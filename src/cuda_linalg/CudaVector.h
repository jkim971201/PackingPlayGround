#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

#include <vector>
#include <cstdio>

#include <cusparse.h>
#include <thrust/device_vector.h>

#include "CudaUtil.h"
#include "CudaObject.h"

namespace cuda_linalg
{

template<typename T>
class CudaVector : public CudaObject
{
  public:

    CudaVector() 
    {
      T* d_data_ptr = thrust::raw_pointer_cast(d_data_.data());
      CHECK_CUSPARSE( cusparseCreateDnVec(&descriptor_, 0, d_data_ptr, CUDA_R_64F) )
    }

    CudaVector(const std::vector<T>& h_vector) 
    {
      size_t n = h_vector.size();
      d_data_.resize(n);

      thrust::copy(h_vector.begin(), h_vector.end(), d_data_.begin());

      T* d_data_ptr = thrust::raw_pointer_cast(d_data_.data());
      CHECK_CUSPARSE( cusparseCreateDnVec(&descriptor_, n, d_data_ptr, CUDA_R_64F) )
    }

    CudaVector(size_t n) 
    { 
      d_data_.resize(n); 
      thrust::fill(d_data_.begin(), d_data_.end(), 0);

      T* d_data_ptr = thrust::raw_pointer_cast(d_data_.data());
      CHECK_CUSPARSE( cusparseCreateDnVec(&descriptor_, n, d_data_ptr, CUDA_R_64F) )
    }

    cusparseDnVecDescr_t getDescriptor()       { return descriptor_; }
    cusparseDnVecDescr_t getDescriptor() const { return descriptor_; }

    size_t size() const { return d_data_.size(); }

          T* data()       { return thrust::raw_pointer_cast(d_data_.data()); }
    const T* data() const { return thrust::raw_pointer_cast(d_data_.data()); }

    // thrust::device_vector<T>::iterator -> auto?
          auto begin()       { return d_data_.begin(); }
    const auto begin() const { return d_data_.begin(); }

          auto end()         { return d_data_.end(); }
    const auto end()   const { return d_data_.end(); }

    thrust::device_vector<T>& getVector() { return d_data_; }

    void resetDescriptor()
    {
      CHECK_CUSPARSE(
        cusparseDnVecSetValues(
          descriptor_, 
          thrust::raw_pointer_cast(d_data_.data())) )
    }

    void swap(CudaVector& vec)
    {
      // swap is allowed between vectors with same size.
      assert(vec.size() == size());

      d_data_.swap(vec.getVector());

      vec.resetDescriptor();
      resetDescriptor();
    }

    void resize(size_t n) 
    { 
      d_data_.clear(); 
      d_data_.resize(n); 

      CHECK_CUSPARSE( cusparseDestroyDnVec(descriptor_) )
      T* d_data_ptr = thrust::raw_pointer_cast(d_data_.data());
      CHECK_CUSPARSE( cusparseCreateDnVec(&descriptor_, n, d_data_ptr, CUDA_R_64F) )
    }

    void clear() 
    { 
      d_data_.clear();
    }

    void fillZero()
    {
      thrust::fill(d_data_.begin(), d_data_.end(), 0);
    }

  private:

    thrust::device_vector<T> d_data_;
    cusparseDnVecDescr_t descriptor_;
};

}

#endif
