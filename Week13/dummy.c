#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  long n;
  long k;
  int i, j, l, s;

  n = strtol(argv[1], NULL, 10);
  k = strtol(argv[2], NULL, 10);

  s = 0;
  for (i = 0; i < n; i++)
      for (j = 0; j < k; j++)
	  for (l = 0; l < k; l++)
	      s = s + 1;
  printf("%d\n", s);
}
