#include <cstdio>
#include <immintrin.h>

/* lade aus Array 8 int ins Vektorregister */
#define LOAD_VEC(arr) _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr))

/* speichere Vektor ins Array */
#define STORE_VEC(arr, vec)                                               \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec)

/* Blend-Maske aus Permutations-Indices für aufsteigendes Sortieren */
#define ASC(a, b, c, d, e, f, g, h)                                       \
  ((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) |     \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0)

/* Blend-Maske aus Permutations-Indices für absteigendes Sortieren */
#define DESC(a, b, c, d, e, f, g, h)                                      \
  ((h > 6) << 7) | ((g > 5) << 6) | ((f > 4) << 5) | ((e > 3) << 4) |     \
      ((d > 2) << 3) | ((c > 1) << 2) | ((b > 0) << 1) | (a > -1)

/* vektorisiertes compare-exchange mit Permutation */
#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){                  \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);     \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask);    \
    __m256i min = _mm256_min_epi32(permuted, vec);                        \
    __m256i max = _mm256_max_epi32(permuted, vec);                        \
    vec = _mm256_blend_epi32(min, max, MASK(a, b, c, d, e, f, g, h));}

/* vektorisiertes compare-exchange mit Shuffle */
#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){                  \
    constexpr auto shuffle_mask = _MM_SHUFFLE(d, c, b, a);                \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);           \
    __m256i min = _mm256_min_epi32(shuffled, vec);                        \
    __m256i max = _mm256_max_epi32(shuffled, vec);                        \
    vec = _mm256_blend_epi32(min, max, MASK(a, b, c, d, e, f, g, h));}

/* Sortiernetzwerk für 8 int mit compare-exchange Makros */
#define SORT_8(vec, ASC_OR_DESC){                                         \
    COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC_OR_DESC);               \
    COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC_OR_DESC);               \
    COEX_SHUFFLE(vec, 0, 2, 1, 3, 4, 6, 5, 7, ASC_OR_DESC);               \
    COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, ASC_OR_DESC);               \
    COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC_OR_DESC);               \
    COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC_OR_DESC);}

int main() {
  int arr[8] = {7, 8, 5, 1, 4, 3, 2, 6};
  auto vec = LOAD_VEC(arr);
  SORT_8(vec, ASC); /* sortiere in aufsteigender Reihenfolge */
  STORE_VEC(arr, vec);
  for(int i: arr) printf("%d ", i); /* 1 2 3 4 5 6 7 8 */
  puts("");
  SORT_8(vec, DESC); /* sortiere in absteigender Reihenfolge */
  STORE_VEC(arr, vec);
  for(int i: arr) printf("%d ", i); /* 8 7 6 5 4 3 2 1 */
  return 0;}