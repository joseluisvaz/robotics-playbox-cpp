set(CMAKE_FOLDER "Magnum/imgui_integration/base_application/")

add_library(base_application base_application.cpp base_application.hpp)
target_include_directories(base_application PUBLIC
    ${PROJECT_SOURCE_DIR}/src/)
target_compile_features(base_application PUBLIC cxx_std_17)
target_link_libraries(base_application PUBLIC 
    ImPlot
    Common
    GraphicsObjects
    easy_profiler
    MagnumImGuiIntegration
    Magnum::Sdl2Application
    Magnum::GL 
    Magnum::Shaders
    Magnum::Primitives
    Magnum::SceneGraph
    Magnum::Shaders
    Corrade::Main
    Eigen3::Eigen
    cem_mpc)