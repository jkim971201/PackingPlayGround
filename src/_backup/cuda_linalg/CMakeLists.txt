find_package(CUDAToolkit)
find_package(Thrust)
find_package(Eigen3 REQUIRED CONFIG)

enable_language(CUDA)

set(CUDA_SRC
  CudaObject.cu
  CudaVectorAlgebra.cu
  CudaMatrixAlgebra.cu
  CudaSparseMatrixAlgebra.cu
)

set_source_files_properties(${CUDA_SRC} PROPERTIES LANGUAGE CUDA)

add_library(CudaLinAlg OBJECT ${CUDA_SRC})

set_target_properties(CudaLinAlg PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(CudaLinAlg
	PRIVATE 
  Eigen3::Eigen
)

target_include_directories(CudaLinAlg
  PRIVATE
  Eigen3::Eigen
  ../
)

