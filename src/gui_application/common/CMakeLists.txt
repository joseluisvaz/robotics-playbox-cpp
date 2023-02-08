set(CMAKE_FOLDER "Magnum/imgui_integration/common/")

find_package(Eigen3 3.4.0 REQUIRED)

add_library(Common math.hpp math.cpp types.hpp)
target_compile_features(Common PUBLIC cxx_std_17)
target_include_directories(Common PUBLIC
    ${PROJECT_SOURCE_DIR}/src/)
target_link_libraries(Common PRIVATE
   Eigen3::Eigen 
   easy_profiler)