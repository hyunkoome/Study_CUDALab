cmake_minimum_required(VERSION 3.27)
project(Ex06_NumericalErrors CXX)

# CLion에서 CUDA 인클루드를 쓸 필요가 없으므로 제거 또는 조건 처리
# if ($ENV{CLION_IDE})
#     include_directories("/usr/local/cuda-12.6/include")
# endif ()

set(CMAKE_CXX_STANDARD 17)

# CUDA 관련 설정 제거
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
# set_target_properties(Ex01_CpuMultithreading PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(Ex06_NumericalErrors Ex06_NumericalErrors.cpp)
# pthread 라이브러리 추가 (std::thread 사용 시 필수)
target_link_libraries(Ex06_NumericalErrors pthread)