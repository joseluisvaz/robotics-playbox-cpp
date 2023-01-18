set(CMAKE_FOLDER "Magnum/imgui_integration/cem_mpc/")

set(CEM_MPC_HEADERS cem_mpc.hpp cem_mpc.tpp)
set(CEM_MPC_SRCS cem_mpc.cpp)

add_library(CEM_MPC 
    cem_mpc.cpp
    cem_mpc.hpp
    cem_mpc.tpp
)

target_compile_features(CEM_MPC PUBLIC cxx_std_17)
target_include_directories(CEM_MPC PUBLIC
    ${PROJECT_SOURCE_DIR}/src/gui_application/)
target_link_libraries(CEM_MPC PRIVATE
   Eigen3::Eigen 
   Common
   easy_profiler)

add_executable(cem_mpc_application WIN32 cem_mpc_application.cpp cem_mpc_application.hpp intelligent_driver_model.hpp)
target_link_libraries(cem_mpc_application 
PUBLIC 
    base_application
    Common
    Eigen3::Eigen
)