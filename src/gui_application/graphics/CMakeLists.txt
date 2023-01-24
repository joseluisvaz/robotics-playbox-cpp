find_package(Magnum REQUIRED 
    GL 
    Shaders)
find_package(Corrade REQUIRED Main)

#******** Add  graphics object 
add_library(GraphicsObjects
    graphics_objects.cpp
    graphics_objects.hpp
)
target_compile_features(GraphicsObjects PUBLIC cxx_std_17)

target_include_directories(GraphicsObjects PRIVATE 
    ${PROJECT_SOURCE_DIR}/src/
)
target_link_libraries(GraphicsObjects PUBLIC 
   Corrade::Main
   environment
   Magnum::GL
    Eigen3::Eigen
)
