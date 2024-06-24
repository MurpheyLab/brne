cmake_minimum_required(VERSION 3.9)
project(brnelib)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS " -Ofast -O3 ${SSE_FLAGS} -msse4 -fopenmp")

# A special treat for OpenMP
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message(STATUS "\nFound OpenMP! Turning on support for parallelization\n")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# create the library
add_library(brnelib src/brne.cpp)
target_link_libraries(brnelib ${ARMADILLO_LIBRARIES} OpenMP::OpenMP_CXX)
# enable C++ 17
target_compile_features(brnelib PUBLIC cxx_std_17) 
# warnings are your friend!
target_compile_options(brnelib PUBLIC -Wall -Wextra -Wpedantic)

add_executable(two_pedestrians src/two_pedestrians.cpp)
target_link_libraries(two_pedestrians brnelib)

add_executable(robot_one_ped src/robot_one_ped.cpp)
target_link_libraries(robot_one_ped brnelib)

add_executable(sampling_test src/sampling_test.cpp)
target_link_libraries(sampling_test brnelib)

if(NOT CMAKE_CROSSCOMPILING)
# stuff you don't want to be done when cross compiling
# CMake also has the ability to generate doxygen documentation
find_package(Doxygen)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
doxygen_add_docs(doxygen include/ src/ README.md ALL)
# Use the cmake testing functionality. A test is just an executable.
# We use the catch2 unit testing framework
find_package(Catch2 3 REQUIRED)
include(CTest)
# enable_testing()
add_executable(brne_test tests/brne_test.cpp)
target_link_libraries(brne_test brnelib Catch2::Catch2WithMain)
add_test(NAME Test_of_Brnelib COMMAND brne_test)
endif()

target_include_directories(brnelib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(two_pedestrians PUBLIC -Wall)
target_compile_options(robot_one_ped PUBLIC -Wall)
target_compile_options(sampling_test PUBLIC -Wall)
# install the include directories
install(DIRECTORY include/brnelib DESTINATION include)
# Install the targets and create a CMake Exported Target
install(TARGETS two_pedestrians robot_one_ped sampling_test brnelib EXPORT brnelib-targets)
# The project_name-targets now also needs to be exported.
install(EXPORT brnelib-targets
        FILE brnelib-config.cmake
        NAMESPACE brnelib::
        DESTINATION lib/cmake/${PROJECT_NAME})