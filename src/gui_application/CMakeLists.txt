set(CMAKE_FOLDER "Magnum/imgui_integration")



find_package(Eigen3 3.4.0 REQUIRED)
find_package(Magnum REQUIRED 
    GL 
    Shaders
    Primitives
    SceneGraph
    Shaders
    Sdl2Application)
find_package(Catch2 3 REQUIRED)
find_package(Magnum REQUIRED)
find_package(MagnumIntegration REQUIRED)
find_package(SDL2 REQUIRED)
include_directories(/usr/include/SDL2/)

# Find python and its include files so that we can use matplotlibcpp.h
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
endif()

# GUI test application for quick ability to verify changes w/o having to
# compile examples as well (and to ensure the template APIs don't get out of
# sync with the apps, as the ContextGLTest has only a mock)
find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED Sdl2Application)
find_package(easy_profiler REQUIRED)

#******** Add ImPlot
#END ******** Add ImPlot

# Add common subdirectories
add_subdirectory(graphics)
add_subdirectory(common)
add_subdirectory(third_party)
add_subdirectory(base_application)
# Add applications to showcase algorithms
add_subdirectory(cem_mpc)
add_subdirectory(ilqr_mpc)
add_subdirectory(example_application)
add_subdirectory(tsdf_application)

# Compiles tests for the repo
file(GLOB_RECURSE TEST_FILES "*_test.cpp")
add_executable(roboticssandbox-tests main.cpp ${TEST_FILES} finite_diff.hpp)
target_link_libraries(roboticssandbox-tests 
PUBLIC
    Common
    Catch2::Catch2WithMain
    Eigen3::Eigen)
set_target_properties(roboticssandbox-tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)