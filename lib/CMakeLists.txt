###############################################################
set(HEADERS_PLATE
    Plate/include/plate.hpp
)

set(SOURCES_PLATE
    Plate/src/plate.cpp
)

add_library(plate
    ${HEADERS_PLATE}
    ${SOURCES_PLATE}
)

target_include_directories(plate
    PRIVATE Plate/src
    PUBLIC Plate/include
)

###############################################################
set(HEADERS_SOLVEFE
    SolveFE/include/element_routine.hpp
    SolveFE/include/material_routine.hpp
    SolveFE/include/utility.hpp
    SolveFE/include/solve.hpp
)

set(SOURCES_SOLVEFE
    SolveFE/src/element_routine.cpp
    SolveFE/src/material_routine.cpp
    SolveFE/src/utility.cpp
    SolveFE/src/solve.cpp
)

add_library(solveFE
    ${HEADERS_SOLVEFE}
    ${SOURCES_SOLVEFE}
)

target_include_directories(solveFE
    PRIVATE SolveFE/src
    PUBLIC SolveFE/include
)

###############################################################
