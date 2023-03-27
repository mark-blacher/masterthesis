#include <cstdio>
#include <immintrin.h>

/* vektorisiertes compare-exchange mit Permutation */
#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){                       \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);          \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask);         \
    __m256i min = _mm256_min_epi32(permuted, vec);                             \
    __m256i max = _mm256_max_epi32(permuted, vec);                             \
    vec = _mm256_blend_epi32(min, max, MASK);}


#define SORT_8(vec){                       /* sortiere aufsteigend 8 int */    \
    COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, 0b10101010);  /* Schritt 1 */    \
    COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, 0b11001100);  /* Schritt 2 */    \
    COEX_PERMUTE(vec, 0, 2, 1, 3, 4, 6, 5, 7, 0b01000100);  /* Schritt 3 */    \
    COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, 0b11110000);  /* Schritt 4 */    \
    COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, 0b11001100);  /* Schritt 5 */    \
    COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, 0b10101010);} /* Schritt 6 */

int main() {
  int arr[8] = {7, 8, 5, 1, 4, 3, 2, 6};
  __m256i vec = _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr));
  SORT_8(vec);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec);
  for(int i: arr) printf("%d ", i);} /* 1 2 3 4 5 6 7 8 */