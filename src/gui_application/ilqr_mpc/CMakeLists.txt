set(CMAKE_FOLDER "Magnum/imgui_integration/ilqr_mpc/")

add_executable(ilqr_application ilqr_application.cpp ilqr_application.hpp ilqr_mpc.hpp ilqr_mpc.cpp)
target_include_directories(ilqr_application PUBLIC
    ${PROJECT_SOURCE_DIR}/src/)
target_compile_features(ilqr_application PUBLIC cxx_std_17)
target_link_libraries(ilqr_application 
PUBLIC 
    base_application 
    Eigen3::Eigen
   # Add python libraries so that we can use matplotlibcpp.h for debugging
   ${PYTHON_LIBRARIES}
)