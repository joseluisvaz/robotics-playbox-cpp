
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

# GUI test application for quick ability to verify changes w/o having to
# compile examples as well (and to ensure the template APIs don't get out of
# sync with the apps, as the ContextGLTest has only a mock)
find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED Sdl2Application)

add_library(ImPlot ${SHARED_OR_STATIC} 
    implot.cpp
    implot_items.cpp
    implot.h
    implot_internal.h
)
target_include_directories(ImPlot PUBLIC
    ${PROJECT_SOURCE_DIR}/src/MagnumExternal/ImGui)

add_executable(gui_application WIN32 gui_application.cc gui_application.hh)
target_compile_features(gui_application PRIVATE cxx_std_17)
target_link_libraries(gui_application PRIVATE
    MagnumImGuiIntegration
    ImPlot
    Corrade::Main)
