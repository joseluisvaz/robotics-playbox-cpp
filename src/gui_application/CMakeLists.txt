
set(CMAKE_FOLDER "Magnum/imgui_integration")

find_package(Magnum REQUIRED 
    GL 
    Shaders 
    Primitives
    SceneGraph
    Shaders
    Sdl2Application)
find_package(Magnum REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)

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
#END ******** Add ImPlot

add_executable(gui_application WIN32 gui_application.cc gui_application.hh)
target_compile_features(gui_application PRIVATE cxx_std_17)
target_link_libraries(gui_application PRIVATE
    MagnumImGuiIntegration
    easy_profiler
    ImPlot
    Eigen3::Eigen
    Corrade::Main)


