#include <stdio.h>

/* Makro COEX (compare-exchange) simuliert Vergleichsmodul */
#define COEX(a, b) if(a > b) {int c = a; a = b; b = c;}
/* Sortiernetzwerk für 4 Integer */
inline void sort_4_int(int *arr) {
  COEX(arr[0], arr[1]); COEX(arr[2], arr[3]); /* Schritt 1 */
  COEX(arr[0], arr[2]); COEX(arr[1], arr[3]); /* Schritt 2 */
  COEX(arr[1], arr[2]);}                      /* Schritt 3 */


int main() {
  int arr[4] = {2, -2, 1, -1};
  sort_4_int(arr);
  printf("%d %d %d %d", arr[0], arr[1], arr[2], arr[3]);
}