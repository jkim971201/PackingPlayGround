cmake_minimum_required(VERSION 3.1)

project(Packer VERSION 1 LANGUAGES CXX)

find_package(Qt5 COMPONENTS Core Widgets)

set(CMAKE_C_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SRC
	src/main.cpp
	src/Packer.cpp
	src/Painter.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb -O3")
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb -O3")

add_executable(${PROJECT_NAME}
	${SRC}
)

target_link_libraries(${PROJECT_NAME}
	Qt5::Core
	Qt5::Widgets
)

MESSAGE( STATUS "QT_INCLUDE_PATH: ${QT5_INCLUDE_PATH}")
MESSAGE( STATUS "QT_LIBRARY_PATH: ${QT5_LIBRARY_PATH}")
