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

# #Pybind
# find_package (pybind11 REQUIRED)
# message("Python VERSION: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

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

# #Eigen
find_package (Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message("Eigen3 Found!")
else (Eigen3_FOUND)
    message(SEND_ERROR "This application cannot compile without Eigen3")
endif (Eigen3_FOUND)
include_directories(${EIGEN3_INCLUDE_DIR})

#######################################################################

# #Qt
# message("------------------------------------")
# message("---- LOADING Qt...")
# find_package(Qt5 COMPONENTS Widgets Core Gui PrintSupport REQUIRED)

# if (${Qt5_FOUND})
#     message("-- Qt5_VERSION: ${Qt5_VERSION}")
# else()
#     message("-- Qt5 NOT FOUND!")
# endif()

# # include(${QT_USE_FILE})
# add_definitions(${QT_DEFINITIONS})

# set(QT5_LIBRARIES
#     Qt5::Widgets
#     Qt5::Core
#     Qt5::Gui
#     Qt5::PrintSupport
# )

# include_directories(${QT_INCLUDES})

# get_target_property(QtCore_location Qt5::Core LOCATION)

#######################################################################

#thirdparty
# include_directories(thirdparty/eigen)
add_subdirectory(thirdparty/pybind11)

#######################################################################

#testing
enable_testing()
add_subdirectory(thirdparty/googletest)
add_subdirectory(tests)

#######################################################################

set(CMAKE_C_COMPILER /usr/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/bin/mpiCC)

# set(CMAKE_C_COMPILER /usr/bin/gcc)
# set(CMAKE_CXX_COMPILER /usr/bin/g++)

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
    PRIVATE ${MPI_LIBRARIES}
    PRIVATE plate
    PRIVATE solveFE
)

#######################################################################

if(TARGET Threads::Threads)                                                                                                                                                                                                                   
  target_link_libraries(${This} PRIVATE Threads::Threads)                                                                                                                                                                                       
endif()

#######################################################################
