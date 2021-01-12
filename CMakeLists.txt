cmake_minimum_required(VERSION 3.9)

set(This topoNN)

project(${This} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#######################################################################

#thread
set(CMAKE_THREAD_PREFER_PTHREAD ON)                                                                                                                                                                                                           
set(THREADS_PREFER_PTHREAD_FLAG ON)                                                                                                                                                                                                           
find_package(Threads REQUIRED)

#######################################################################

# #OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
    message("OpenMP Found!")
else (OpenMP_FOUND)
    message(SEND_ERROR "OpenMP NOT Found")
endif(OpenMP_FOUND)

#######################################################################

# #MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI Found!")
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

#######################################################################

#thirdparty
include_directories(thirdparty/eigen)
add_subdirectory(thirdparty/pybind11)

#######################################################################

#testing
enable_testing()
add_subdirectory(thirdparty/googletest)
add_subdirectory(tests)

#######################################################################

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -Wall -O3 -std=c++11")
 
#######################################################################

#sub-directories
add_subdirectory(lib)

#######################################################################

#as a mainfile
add_executable(${This} main.cpp)

#######################################################################

# #as a python module
# pybind11_add_module(${This}
#     wrapper.cpp
# )

#######################################################################

target_link_libraries(${This}
    # PRIVATE pybind11::embed
    PRIVATE plate
    PRIVATE solveFE
)

#######################################################################

if(TARGET Threads::Threads)                                                                                                                                                                                                                   
  target_link_libraries(${This} PRIVATE Threads::Threads)                                                                                                                                                                                       
endif()

#######################################################################