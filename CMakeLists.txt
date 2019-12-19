cmake_minimum_required(VERSION 2.8)
include_directories(sparse_learn/c/)
project(sparse-learn C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3 -fPIC")

add_executable(test_fast_pcst sparse_learn/c/fast_pcst_test.c
        sparse_learn/c/fast_pcst.c sparse_learn/c/fast_pcst.h)
target_link_libraries(test_fast_pcst -Wall -O3 -lm)

add_executable(test_head_tail_proj sparse_learn/c/head_tail_proj_test.c
        sparse_learn/c/head_tail_proj.c sparse_learn/c/head_tail_proj.h)
target_link_libraries(test_head_tail_proj -Wall -O3 -lm)

add_executable(test_sort sparse_learn/c/sort_test.c sparse_learn/c/sort.c sparse_learn/c/sort.h)
target_link_libraries(test_sort -Wall -O3 -lm)