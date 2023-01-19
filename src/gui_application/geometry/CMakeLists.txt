set(CMAKE_FOLDER "Magnum/imgui_integration/geometry/")

find_package(Eigen3 3.4.0 REQUIRED)

add_library(geometry geometry.cpp geometry.hpp)
target_compile_features(geometry  PUBLIC cxx_std_17)
target_include_directories(geometry PUBLIC
    ${PROJECT_SOURCE_DIR}/src/gui_application/)
target_link_libraries(geometry 
PRIVATE 
    Eigen3::Eigen
)