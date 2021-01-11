###############################################################
set(PLATE_HEADERS
    Plate/include/plate_rect.hpp
)

set(PLATE_SOURCES
    Plate/src/plate_rect.cpp
)

add_library(plate SHARED
    ${PLATE_HEADERS}
    ${PLATE_SOURCES}
)

target_include_directories(plate
    PRIVATE Plate/src
    PUBLIC Plate/include
)

###############################################################
set(SOLVEFE_HEADERS
    SolveFE/include/element_routine.hpp
    SolveFE/include/material_routine.hpp
    SolveFE/include/utility.hpp
    SolveFE/include/solve.hpp
)

set(SOLVEFE_SOURCES
    SolveFE/src/element_routine.cpp
    SolveFE/src/material_routine.cpp
    SolveFE/src/utility.cpp
    SolveFE/src/solve.cpp
)

add_library(solveFE SHARED
    ${SOLVEFE_HEADERS}
    ${SOLVEFE_SOURCES}
)

target_include_directories(solveFE
    PRIVATE SolveFE/src
    PUBLIC SolveFE/include
)

###############################################################
