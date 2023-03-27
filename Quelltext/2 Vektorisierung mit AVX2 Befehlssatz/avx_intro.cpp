#include <immintrin.h>
#include <iostream>

using namespace std;

void print_m256i(__m256i &v) {
  for (int i = 0; i < 8; ++i) {
    cout << reinterpret_cast<int *>(&v)[i] << " ";
  }
  cout << endl;
}

void print_separator(){cout << "------------------------------\n";};

int main() {
  int arr[8] = {3, 4, 15, 6, 16, 10, 5, 8};
  /* lade Vektor von einer nicht ausgerichteten Speicheradresse */
  __m256i v_loaded = _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr));
  /* erstelle Vektor mit acht unterschiedlichen Integern */
  __m256i v_different = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
  /* erstelle Vektor mit acht gleichen Integern */
  __m256i v_same = _mm256_set1_epi32(7);

  print_m256i(v_loaded);
  print_m256i(v_different);
  print_m256i(v_same);
  print_separator();

  __m256i indexes = _mm256_setr_epi32(0, 1, 6, 3, 7, 5, 2, 4);
  /* v_loaded permutiert von <3 4 15 6 16 10 5 8> zu <3 4 5 6 8 10 15 16> */
  v_loaded = _mm256_permutevar8x32_epi32(v_loaded, indexes);
  /* v_different shuffelt von <1 2 3 4 5 6 7 8> zu <4 3 2 1 8 7 6 5> */
  v_different = _mm256_shuffle_epi32(v_different, 0b00011011);

  print_m256i(v_loaded);
  print_m256i(v_different);
  print_separator();

  __m256i v1 = _mm256_setr_epi32(1, 10, 11, 4, 5, 14, 7, 16);
  __m256i v2 = _mm256_setr_epi32(9, 2, 3, 12, 13, 6, 15, 8);
  /* v_blended ist <1 2 3 4 5 6 7 8> */
  __m256i v_blended = _mm256_blend_epi32(v1, v2, 0b10100110);

  print_m256i(v_blended);
  print_separator();

  /* v_min ist <1 2 3 4 5 6 7 8> */
  __m256i v_min = _mm256_min_epi32(v1, v2);
  /* v_max ist <9 10 11 12 13 14 15 16> */
  __m256i v_max = _mm256_max_epi32(v1, v2);

  print_m256i(v_min);
  print_m256i(v_max);
  print_separator();

  /* v_loaded nach Array arr speichern,
   * Speicheradresse muss nicht ausgerichtet sein */
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), v_loaded);
  /* Array arr ist <3 4 5 6 8 10 15 16> */
  for (int i = 0; i < 8; ++i) {cout << arr[i] << " ";};
}