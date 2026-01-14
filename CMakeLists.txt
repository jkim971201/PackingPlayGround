cmake_minimum_required(VERSION 3.1)

project(MacroPlacer LANGUAGES CXX)

add_subdirectory(src/cuda_linalg)

find_package(Eigen3 REQUIRED CONFIG)
find_package(CUDAToolkit)
find_package(Thrust)
find_package(Qt5 COMPONENTS Core Widgets)

enable_language(CUDA)

set(CMAKE_C_COMPILER "g++")
set(CMAKE_CXX_STANDARD 20)

set(SRC
	src/main.cpp
  src/FileIO.cpp
  src/MacroPlacer.cpp
	src/Painter.cpp
  src/Util.cpp
  src/SDPSolverCPU.cpp
  src/SDPSolverGPU.cu
  src/objects/Pin.cpp
  src/objects/Net.cpp
  src/objects/Macro.cpp
)

set_source_files_properties(src/SDPSolverGPU.cu PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME}
  PUBLIC
		${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /home/jkim/mosek/10.1/tools/platform/linux64x86/h
)

target_link_libraries(${PROJECT_NAME}
	Qt5::Core
	Qt5::Widgets
  Eigen3::Eigen
  /home/jkim/mosek/10.1/tools/platform/linux64x86/bin/libmosek64.so.10.1
  /home/jkim/mosek/10.1/tools/platform/linux64x86/bin/libfusion64.so.10.1

  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
  CUDA::cufft
  CUDA::cusolver
  CudaLinAlg
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75")
