
set(CMAKE_FOLDER "Magnum/imgui_integration")

add_subdirectory(graphics)

find_package(Magnum REQUIRED 
    GL 
    Shaders 
    Primitives
    SceneGraph
    Shaders
    Sdl2Application)
find_package(Catch2 3 REQUIRED)
find_package(Magnum REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)


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
add_library(ImPlot ${SHARED_OR_STATIC} 
    implot.cpp
    implot_items.cpp
    implot.h
    implot_internal.h
)
target_include_directories(ImPlot PUBLIC 
    ${PROJECT_SOURCE_DIR}/src/MagnumExternal/ImGui)

# Diable all warnings from this external library.
target_compile_options(ImPlot PUBLIC -w)
#END ******** Add ImPlot

#******** Add CEM_MPC 
set(CEM_MPC_HEADERS cem_mpc.hpp types.hpp math.hpp cem_mpc.tpp)
set(CEM_MPC_SRCS cem_mpc.cpp math.cpp)
add_library(CEM_MPC ${SHARED_OR_STATIC} 
    ${CEM_MPC_SRCS}
    ${CEM_MPC_HEADERS}
)
target_compile_features(CEM_MPC PUBLIC cxx_std_17)
target_include_directories(CEM_MPC PUBLIC
    ${PROJECT_SOURCE_DIR}/src/gui_application/)
target_link_libraries(CEM_MPC PRIVATE
   Eigen3::Eigen 
   easy_profiler)
#END ******** Add CEM_MPC 

add_library(base_example base_example.cpp base_example.hpp types.hpp math.hpp)
target_compile_features(base_example PUBLIC cxx_std_17)
target_link_libraries(base_example PUBLIC 
    MagnumImGuiIntegration
    easy_profiler
    ImPlot
    Eigen3::Eigen
    CEM_MPC
    GraphicsObjects
    Corrade::Main)

# add_executable(cem_mpc_example WIN32 cem_mpc_example.cpp cem_mpc_example.hpp types.hpp math.hpp intelligent_driver_model.hpp)
# target_link_libraries(cem_mpc_example 
# PUBLIC 
#     base_example
#     Eigen3::Eigen
# )
# 
# add_executable(template_example WIN32 template_example.cpp template_example.hpp)
# target_link_libraries(template_example 
# PUBLIC 
#     base_example
# )
# 
# add_executable(tsdf_example WIN32 tsdf_example.cpp tsdf_example.hpp)
# target_link_libraries(tsdf_example 
# PUBLIC 
#     base_example
# )

add_executable(ilqr_main WIN32 ilqr_main.cpp ilqr_main.hpp ilqr_mpc.hpp ilqr_mpc.cpp)
target_link_libraries(ilqr_main 
PUBLIC 
   base_example 
   # Add python libraries so that we can use matplotlibcpp.h for debugging
   ${PYTHON_LIBRARIES}
)

file(GLOB_RECURSE TEST_FILES "*_test.cpp")
add_executable(roboticssandbox-tests main.cpp ${TEST_FILES} finite_diff.hpp types.hpp)
target_link_libraries(roboticssandbox-tests 
PUBLIC
    Catch2::Catch2WithMain
    Eigen3::Eigen)
set_target_properties(roboticssandbox-tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    #autodiff::autodiff
    