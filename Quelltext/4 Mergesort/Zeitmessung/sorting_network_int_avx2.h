#ifndef HEADER_NETWORKS_SORTING_NETWORK_INT_AVX2_H
#define HEADER_NETWORKS_SORTING_NETWORK_INT_AVX2_H

#include <immintrin.h>

/* buffer must be aligned to 32 byte */
void sort_int_sorting_network(int* arr, int* buffer, int n);


/* if array is 32 byte aligned and n % 16 == 0 no buffer is needed */
void sort_int_sorting_network_aligned(int* arr, int n);


/* if array is 32 byte aligned and N % 2 == 0 no buffer is needed */
void sort_int_sorting_network_aligned(__m256i* vecs, int N);


#endif //HEADER_NETWORKS_SORTING_NETWORK_INT_AVX2_H
