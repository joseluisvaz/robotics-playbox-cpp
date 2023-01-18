set(CMAKE_FOLDER "Magnum/imgui_integration/third_party/implot/")

add_library(ImPlot 
    implot.cpp
    implot_items.cpp
    implot.h
    implot_internal.h
)
target_include_directories(ImPlot PUBLIC 
    ${PROJECT_SOURCE_DIR}/src/MagnumExternal/ImGui)

# Diable all warnings from this external library.
target_compile_options(ImPlot PUBLIC -w)