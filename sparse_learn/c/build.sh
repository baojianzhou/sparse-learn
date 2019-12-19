#!/bin/bash
sort_src="sort.h sort.c"
fast_pcst_src="fast_pcst.h fast_pcst.c"
head_tail_src="head_tail_proj.h head_tail_proj.c"
all_src="${sort_src} ${fast_pcst_src} ${head_tail_src}"
gcc -g -Wall -std=c11 -O3 ${sort_src} sort_test.c -o test_sort
gcc -g -Wall -std=c11 -O3 ${fast_pcst_src} fast_pcst_test.c -o test_fast_pcst
gcc -g -Wall -std=c11 -O3 ${all_src} head_tail_proj_test.c -o test_head_tail_proj -lm