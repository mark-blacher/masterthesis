#include <cstdio>
#include <immintrin.h>

#define COEX_VERTICAL_128i(a, b){ /* berechne vier Vergleichsmodule */    \
  __m128i c = a; a = _mm_min_epi32(a, b); b = _mm_max_epi32(c, b);}

/* shuffle 2 Vektoren, Instruktion für int fehlt, deshalb mit float */
#define SHUFFLE_TWO_VECS(a, b, mask)                                      \
  reinterpret_cast<__m128i>(_mm_shuffle_ps(                               \
      reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b), mask));

int main() {
  int arr[8] = {11, 29, 13, 23, 37, 17, 19, 31};    /* lade Vektoren */
  __m128i v1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(arr));
  __m128i v2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(arr + 4));

  COEX_VERTICAL_128i(v1, v2);                           /* Schritt 1 */

  v2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));  /* Schritt 2 */
  COEX_VERTICAL_128i(v1, v2);

  auto tmp = v1;                                        /* Schritt 3 */
  v1 = SHUFFLE_TWO_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11011101);
  COEX_VERTICAL_128i(v1, v2);

  v2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3));  /* Schritt 4 */
  COEX_VERTICAL_128i(v1, v2);

  tmp = v1;                                             /* Schritt 5 */
  v1 = SHUFFLE_TWO_VECS(v1, v2, 0b01000100);
  v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11101110);
  COEX_VERTICAL_128i(v1, v2);

  tmp = v1;                                             /* Schritt 6 */
  v1 = SHUFFLE_TWO_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11011101);
  COEX_VERTICAL_128i(v1, v2);

  /* Reihenfolge wiederherstellen */
  tmp = _mm_shuffle_epi32(v1, _MM_SHUFFLE(2, 3, 0, 1));
  auto tmp2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));
  v2 = _mm_blend_epi32(tmp, v2, 0b00001010);
  v1 = _mm_blend_epi32(v1, tmp2, 0b00001010);

  /* speichere Vektoren */
  _mm_storeu_si128(reinterpret_cast<__m128i* >(arr), v1);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(arr + 4), v2);
  for (int i = 0; i < 4; ++i) printf("%d ", arr[i]);      /* Ausgabe */
  puts("");
  for (int i = 4; i < 8; ++i) printf("%d ", arr[i]);}