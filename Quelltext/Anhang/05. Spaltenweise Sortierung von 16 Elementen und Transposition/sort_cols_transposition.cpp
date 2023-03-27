#include <cstdio>
#include <immintrin.h>
#include <stdlib.h>

#define COEX_VERTICAL(a, b){  /* berechne acht Vergleichsmodule */ 	     \
  __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b);}

/* transponiere 8 mal 8 Matrix mit Integern */
inline void transpose_8x8_int(__m256i *vecs) {
  __m256 *v = reinterpret_cast<__m256 *>(vecs);
  __m256 a = _mm256_unpacklo_ps(v[0], v[1]);
  __m256 b = _mm256_unpackhi_ps(v[0], v[1]);
  __m256 c = _mm256_unpacklo_ps(v[2], v[3]);
  __m256 d = _mm256_unpackhi_ps(v[2], v[3]);
  __m256 e = _mm256_unpacklo_ps(v[4], v[5]);
  __m256 f = _mm256_unpackhi_ps(v[4], v[5]);
  __m256 g = _mm256_unpacklo_ps(v[6], v[7]);
  __m256 h = _mm256_unpackhi_ps(v[6], v[7]);
  auto tmp = _mm256_shuffle_ps(a, c, 0x4E);
  a = _mm256_blend_ps(a, tmp, 0xCC);
  c = _mm256_blend_ps(c, tmp, 0x33);
  tmp = _mm256_shuffle_ps(b, d, 0x4E);
  b = _mm256_blend_ps(b, tmp, 0xCC);
  d = _mm256_blend_ps(d, tmp, 0x33);
  tmp = _mm256_shuffle_ps(e, g, 0x4E);
  e = _mm256_blend_ps(e, tmp, 0xCC);
  g = _mm256_blend_ps(g, tmp, 0x33);
  tmp = _mm256_shuffle_ps(f, h, 0x4E);
  f = _mm256_blend_ps(f, tmp, 0xCC);
  h = _mm256_blend_ps(h, tmp, 0x33);
  v[0] = _mm256_permute2f128_ps(a, e, 0x20);
  v[1] = _mm256_permute2f128_ps(c, g, 0x20);
  v[2] = _mm256_permute2f128_ps(b, f, 0x20);
  v[3] = _mm256_permute2f128_ps(d, h, 0x20);
  v[4] = _mm256_permute2f128_ps(a, e, 0x31);
  v[5] = _mm256_permute2f128_ps(c, g, 0x31);
  v[6] = _mm256_permute2f128_ps(b, f, 0x31);
  v[7] = _mm256_permute2f128_ps(d, h, 0x31);}

#define CV(a, b) COEX_VERTICAL(a, b) /* Alias für weniger Code */

/* sortiere in 8 Spalten jeweils 16 int mit 60 Modulen */
inline void sort_16_int_vertical(__m256i* vecs){
  CV(vecs[0], vecs[1]); CV(vecs[2], vecs[3]);  /* Schritt 1 */
  CV(vecs[4], vecs[5]); CV(vecs[6], vecs[7]);
  CV(vecs[8], vecs[9]); CV(vecs[10], vecs[11])
  CV(vecs[12], vecs[13]); CV(vecs[14], vecs[15])
  CV(vecs[0], vecs[2]); CV(vecs[1], vecs[3]);  /* Schritt 2 */
  CV(vecs[4], vecs[6]); CV(vecs[5], vecs[7]);
  CV(vecs[8], vecs[10]); CV(vecs[9], vecs[11]);
  CV(vecs[12], vecs[14]); CV(vecs[13], vecs[15]);
  CV(vecs[0], vecs[4]); CV(vecs[1], vecs[5]);  /* Schritt 3 */
  CV(vecs[2], vecs[6]); CV(vecs[3], vecs[7]);
  CV(vecs[8], vecs[12]); CV(vecs[9], vecs[13]);
  CV(vecs[10], vecs[14]); CV(vecs[11], vecs[15]);
  CV(vecs[0], vecs[8]); CV(vecs[1], vecs[9])   /* Schritt 4 */
  CV(vecs[2], vecs[10]); CV(vecs[3], vecs[11])
  CV(vecs[4], vecs[12]); CV(vecs[5], vecs[13])
  CV(vecs[6], vecs[14]); CV(vecs[7], vecs[15])
  CV(vecs[5], vecs[10]); CV(vecs[6], vecs[9]); /* Schritt 5 */
  CV(vecs[3], vecs[12]); CV(vecs[7], vecs[11]);
  CV(vecs[13], vecs[14]); CV(vecs[4], vecs[8]);
  CV(vecs[1], vecs[2]);
  CV(vecs[1], vecs[4]); CV(vecs[7], vecs[13]); /* Schritt 6 */
  CV(vecs[2], vecs[8]); CV(vecs[11], vecs[14]);
  CV(vecs[2], vecs[4]); CV(vecs[5], vecs[6]);  /* Schritt 7 */
  CV(vecs[9], vecs[10]); CV(vecs[11], vecs[13]);
  CV(vecs[3], vecs[8]); CV(vecs[7], vecs[12]);
  CV(vecs[3], vecs[5]); CV(vecs[6], vecs[8]);  /* Schritt 8 */
  CV(vecs[7], vecs[9]); CV(vecs[10], vecs[12]);
  CV(vecs[3], vecs[4]); CV(vecs[5], vecs[6]);  /* Schritt 9 */
  CV(vecs[7], vecs[8]); CV(vecs[9], vecs[10]);
  CV(vecs[11], vecs[12]);
  CV(vecs[6], vecs[7]); CV(vecs[8], vecs[9]); /* Schritt 10 */}

#define SWAP(a, b) {auto vec_tmp = a; a = b; b = vec_tmp;}

int main(){
  __m256i vecs[16];     /* initialisiere Vektoren mit Zufallszahlen */
  int* arr = reinterpret_cast<int*>(vecs);
  for (int i = 0; i < 128; ++i) arr[i] = rand() % 100;

  sort_16_int_vertical(vecs);              /* sortiere spaltenweise */
  transpose_8x8_int(vecs);          /* transponiere ersten 64 Werte */
  transpose_8x8_int(vecs + 8);     /* transponiere letzten 64 Werte */
  /* vertausche Vektoren damit 16 sortierte Elemente in jeder Zeile */
  SWAP(vecs[1], vecs[8]); SWAP(vecs[3], vecs[10]);
  SWAP(vecs[5], vecs[12]); SWAP(vecs[7], vecs[14]);

  for (int i = 0; i < 128; i+=16){           /* zeilenweise Ausgabe */
    for (int j = 0; j < 16; ++j){
      printf("%d ", arr[i + j]);} /* einzelnen Zeilen sind sortiert */
    puts(""); }}
