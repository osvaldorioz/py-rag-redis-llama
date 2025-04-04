# cmake .. -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir) -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ..

cmake_minimum_required(VERSION 3.12)
project(RAGRedis)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED HINTS 
    "${Python3_SITELIB}/pybind11/share/cmake/pybind11" 
    "/usr/local/share/cmake/pybind11"
)
find_library(HIREDIS_LIBRARY NAMES hiredis HINTS /usr/lib /usr/local/lib)
find_library(REDIS_PLUS_PLUS_LIBRARY NAMES redis++ HINTS /usr/local/lib)
find_package(faiss REQUIRED)
find_package(OpenMP REQUIRED)

add_library(rag_module MODULE bindings.cpp)
target_include_directories(rag_module PRIVATE 
    /usr/local/include
    /usr/local/include/faiss  # Asegura que FAISS esté incluido
)
target_link_libraries(rag_module PRIVATE 
    pybind11::module 
    ${Python3_LIBRARIES}
    ${HIREDIS_LIBRARY}
    ${REDIS_PLUS_PLUS_LIBRARY}
    faiss
    OpenMP::OpenMP_CXX
)
set_target_properties(rag_module PROPERTIES PREFIX "" SUFFIX ".so")