cmake_minimum_required(VERSION 3.9)

set(This topoNN)

project(${This})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

###############################################################
find_package(OpenMP)
if (OpenMP_FOUND)
    message("OpenMP Found!")
else (OpenMP_FOUND)
    message(SEND_ERROR "OpenMP NOT Found")
endif(OpenMP_FOUND)
###############################################################
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI Found!")
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)
###############################################################
# set( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
find_package (Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message("Eigen3 Found!")
else (Eigen3_FOUND)
    message(SEND_ERROR "This application cannot compile without Eigen3")
endif (Eigen3_FOUND)
include_directories(${EIGEN3_INCLUDE_DIR})
###############################################################

enable_testing()
add_subdirectory(thirdparty/googletest)
add_subdirectory(tests)

###############################################################
set(CMAKE_C_COMPILER /usr/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/bin/mpiCC)

# set(CMAKE_C_COMPILER /usr/bin/gcc)
# set(CMAKE_CXX_COMPILER /usr/bin/g++)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -Wall -O3 -std=c++11")
 
###############################################################

add_subdirectory(lib)

add_executable(${This} main.cpp)

###############################################################

target_link_libraries(${This} 
    PRIVATE plate
    PRIVATE solveFE
    ${MPI_LIBRARIES}
    OpenMP::OpenMP_CXX
)

###############################################################
