cmake_minimum_required(VERSION 3.9)

###############################################################

set(TESTSOURCES_PLATE
    test_Plate/test_plate_rect.cpp
)

add_executable(test_plate
    ${TESTSOURCES_PLATE}
)

target_link_libraries(test_plate 
    PUBLIC gtest_main
    PUBLIC plate
)

add_test(
    NAME test_plate
    COMMAND test_plate
)

###############################################################

set(TESTSOURCES_SOLVEFE
    test_SolveFE/test_solve.cpp
    test_SolveFE/test_element_routine.cpp
    test_SolveFE/test_material_routine.cpp
)


add_executable(test_solveFE
    ${TESTSOURCES_SOLVEFE}
)


target_link_libraries(test_solveFE
    PUBLIC gtest_main
    PUBLIC solveFE
)

add_test(
    NAME test_solveFE
    COMMAND test_solveFE
)

###############################################################

# set(TESTSOURCES_TOPONN
    
# )


# add_executable(test_topoNN
#     ${TESTSOURCES_TOPONN}
# )


# target_link_libraries(test_topoNN PUBLIC
# 	gtest_main
# )

# add_test(
#     NAME test_topoNN
#     COMMAND test_topoNN
# )

###############################################################


