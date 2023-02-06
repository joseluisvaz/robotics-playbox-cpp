set(CMAKE_FOLDER "Magnum/imgui_integration/cem_mpc/")

##################################################
#   Find CUDA and set compiler flags
##################################################

find_package(CUDA)
message(ERROR this_message)
if(CUDA_FOUND)
    message(ERROR this_message_other)
    enable_language(CUDA)
    include(FindCUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -std=c++17
        -maxrregcount=80;
        -O3;
        -arch=sm_80;
        -gencode=arch=compute_80,code=sm_80;
        -w)
ENDIF()


##################################################
#  Compile cem_mpc library
##################################################

cuda_add_library(cem_mpc 
    cem_mpc.cpp
    cem_mpc.h
    cem_mpc_inl.h
)

target_compile_features(cem_mpc PUBLIC cxx_std_17)
target_include_directories(cem_mpc PUBLIC
    ${PROJECT_SOURCE_DIR}/src/gui_application/
    ${CUDA_LIBRARIES})
target_link_libraries(cem_mpc 
   Eigen3::Eigen 
   Common
   ${CUDA_LIBRARIES}
   easy_profiler)

##################################################
#  Compile cem_mpc application for visualizations 
##################################################

cuda_add_executable(cem_mpc_application cem_mpc_application.cpp cem_mpc_application.hpp intelligent_driver_model.hpp)
target_include_directories(cem_mpc_application PUBLIC
    ${PROJECT_SOURCE_DIR}/src/gui_application/)
target_link_libraries(cem_mpc_application 
        base_application
        Common
        Eigen3::Eigen
        geometry
        environment
    )

##################################################
#  Compile cem_mpc application for racing 
##################################################

cuda_add_executable(cem_mpc_racing_app cem_mpc_racing_app.cpp cem_mpc_racing_app.hpp intelligent_driver_model.hpp)
target_include_directories(cem_mpc_racing_app PUBLIC
    ${PROJECT_SOURCE_DIR}/src/gui_application/)
target_link_libraries(cem_mpc_racing_app 
        base_application
        Common
        Eigen3::Eigen
        geometry
        environment
    )