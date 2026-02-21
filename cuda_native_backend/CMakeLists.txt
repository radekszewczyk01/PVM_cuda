cmake_minimum_required(VERSION 3.18)
project(pvm_cuda LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Default build type ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── CUDA architecture ────────────────────────────────────────────────────────
# Auto-detect GPU on the build machine; fall back to 7.0+ for portability
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Supports sm_70 (Volta) and newer; add 80, 86, 90 as needed
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV      REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# nlohmann/json – embedded single-header (third_party/nlohmann/json.hpp)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ── Sources ──────────────────────────────────────────────────────────────────
set(PVM_SOURCES
    src/pvm_graph.cpp
    src/pvm_kernels.cu
    src/pvm_object.cu
    src/data_provider.cpp
    src/training_manager.cpp
    src/main.cpp
)

add_executable(pvm ${PVM_SOURCES})

# ── Include paths ─────────────────────────────────────────────────────────────
target_include_directories(pvm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

# ── Compile options ───────────────────────────────────────────────────────────
target_compile_options(pvm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler=-march=native
        --expt-relaxed-constexpr
        -lineinfo
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-G -g>
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-g -O0>
    >
)

# ── Linker ────────────────────────────────────────────────────────────────────
target_link_libraries(pvm PRIVATE
    CUDA::cudart
    CUDA::cublas
    ${OpenCV_LIBS}
    nlohmann_json
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS pvm DESTINATION bin)
