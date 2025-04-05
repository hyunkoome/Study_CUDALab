cmake_minimum_required(VERSION 3.27)
project(Ex13_AddingLargeVectors_Serial CUDA)

# RTX 4090 아키텍처 (Ada Lovelace, SM 89)
set(Ex13_AddingLargeVectors_Serial 89)

# CUDA 표준 설정
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CLion에서 CUDA 경로 명시 (필요 시)
if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.6/include")
endif ()

# 실행 파일
add_executable(Ex13_AddingLargeVectors_Serial Ex13_AddingLargeVectors_Serial.cu)

# CUDA 분리 컴파일 옵션
set_target_properties(
    Ex13_AddingLargeVectors_Serial
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared  # 기본값이지만 명시하면 좋음
)

# cuBLAS 라이브러리 추가
target_link_libraries(Ex13_AddingLargeVectors_Serial PRIVATE cublas)
target_link_libraries(Ex13_AddingLargeVectors_Serial PRIVATE dl)

# Optional: 컴파일러 경고 표시 (디버깅에 유용)
target_compile_options(Ex13_AddingLargeVectors_Serial PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
