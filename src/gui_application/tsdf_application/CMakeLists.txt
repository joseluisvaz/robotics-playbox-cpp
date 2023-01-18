set(CMAKE_FOLDER "Magnum/imgui_integration/tsdf_application/")

add_executable(tsdf_application WIN32 tsdf_application.cpp tsdf_application.hpp)
target_link_libraries(tsdf_application 
PUBLIC 
    base_application
)