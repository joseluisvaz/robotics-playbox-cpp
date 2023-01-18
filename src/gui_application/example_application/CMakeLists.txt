set(CMAKE_FOLDER "Magnum/imgui_integration/base_application/")

add_executable(example_application example_application.cpp example_application.hpp)
target_link_libraries(example_application 
PRIVATE 
    base_application
)