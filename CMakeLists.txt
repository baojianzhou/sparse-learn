cmake_minimum_required(VERSION 2.8)
include_directories(sparse_learn/c/)
project(sparse-learn C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra -std=c11 -O3 -fPIC")

set(PYTHON_INCLUDE "/home/baojian/anaconda3/include/python3.7m")
set(NUMPY_INCLUDE "/home/baojian/anaconda3/lib/python3.7/site-packages/numpy/core/include/")
set(PYTHON_LIB "/home/baojian/anaconda3/lib/")

include_directories(${PYTHON_INCLUDE})
set(C_PATH sparse_learn/c/)
set(FAST_PCST_SRC ${C_PATH}fast_pcst.c ${C_PATH}fast_pcst.h)
set(HEAD_TAIL_SRC ${C_PATH}head_tail_proj.c ${C_PATH}head_tail_proj.h)
set(SORT_SRC ${C_PATH}sort.c ${C_PATH}sort.h)

add_executable(test_fast_pcst ${C_PATH}fast_pcst_test.c ${FAST_PCST_SRC})
target_link_libraries(test_fast_pcst -Wall -O3 -lm)

add_executable(test_head_tail_proj ${C_PATH}head_tail_proj_test.c ${HEAD_TAIL_SRC} ${FAST_PCST_SRC} ${SORT_SRC})
target_link_libraries(test_head_tail_proj -Wall -O3 -lm)

add_executable(test_sort ${C_PATH}sort_test.c ${C_PATH}sort.c ${C_PATH}sort.h)
target_link_libraries(test_sort -Wall -O3 -lm)

add_library(sparse_learn SHARED ${C_PATH}main_wrapper.c ${FAST_PCST_SRC} ${HEAD_TAIL_SRC} ${SORT_SRC})
target_link_libraries(sparse_learn -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE} -L${OPENBLAS_LIB} -lm -lpthread)