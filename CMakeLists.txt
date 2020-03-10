# cmake_minimum_required(VERSION <specify CMake version here>)

cmake_minimum_required(VERSION 3.15)
project(PF)

set(CMAKE_CXX_STANDARD 14)

add_executable(PF main.cpp PF.h PF.cpp)

include_directories(D:\\Downloads\\eigen-3.3.7)