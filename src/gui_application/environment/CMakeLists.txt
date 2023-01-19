
set(CMAKE_FOLDER "Magnum/imgui_integration/environment/")


add_library(environment 
    lane_map.cpp
    lane_map.hpp
)

target_compile_features(environment PUBLIC cxx_std_17)
target_include_directories(environment PUBLIC
    ${PROJECT_SOURCE_DIR}/src/gui_application/environment)
target_link_libraries(environment PRIVATE
   Eigen3::Eigen)
