cmake_minimum_required(VERSION 3.1)

project(MacroPlacer LANGUAGES CXX)

find_package(Qt5 COMPONENTS Core Widgets)

set(CMAKE_C_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)

set(SRC
	src/main.cpp
  src/MacroPlacer.cpp
	src/Painter.cpp
  src/objects/Pin.cpp
  src/objects/Net.cpp
  src/objects/Macro.cpp
)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME}
	Qt5::Core
	Qt5::Widgets
)
