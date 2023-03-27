cmake_minimum_required(VERSION 3.7)
project(avx2sort LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)

set(CID ${CMAKE_CXX_COMPILER_ID})

IF (WIN32)
    if ("${CID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s -O2 -std=c++14 -Wall -mavx2 -fopenmp -static-libstdc++")
    elseif ("${CID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Qstd=c++14 /Wall /arch:avx2 /Qdiag-disable:11074,11075 /Qopenmp")
    endif ()
ELSE()
    if ("${CID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++14 -Wall -mavx2 -qopenmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s -O2 -std=c++14 -Wall -mavx2 -fopenmp -static-libstdc++")
    endif ()
ENDIF()

add_executable(avx2sort avx2sort.h main.cpp)