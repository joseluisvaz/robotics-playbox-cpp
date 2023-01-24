set(CMAKE_FOLDER "Magnum/imgui_integration/IterativeLinearQuadraticRegulator/")

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

cuda_add_executable(ilqr_application ilqr_application.cpp ilqr_application.hpp ilqr_mpc.hpp ilqr_mpc.cpp)
target_include_directories(ilqr_application PUBLIC
    ${PROJECT_SOURCE_DIR}/src/
    ${CUDA_LIBRARIES}
    )
target_compile_features(ilqr_application PUBLIC cxx_std_17)
target_link_libraries(ilqr_application 
    base_application 
    Eigen3::Eigen
   # Add python libraries so that we can use matplotlibcpp.h for debugging
   ${CUDA_LIBRARIES}
   ${PYTHON_LIBRARIES}
)