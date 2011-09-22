#include <stdio.h>
#include <limits>
//#include "itkNumericTraits.h"

int main ()
{
   printf ("Hello world\n");

   short min_short = std::numeric_limits<short>::min();
   short max_short = std::numeric_limits<short>::max();
   unsigned short min_ushort = std::numeric_limits<unsigned short>::min();
   unsigned short max_ushort = std::numeric_limits<unsigned short>::max();

   printf ("short %d %d\n", min_short, max_short);
   printf ("ushort %d %d\n", min_ushort, max_ushort);

   short s1 = -33;
   printf ("%d < %d == %s\n", s1, min_ushort, 
       (s1 < min_ushort) ? "true" : "false");

   unsigned short u1 = 44000;
   printf ("%d > %d == %s\n", u1, max_short, 
       (u1 > max_short) ? "true" : "false");
}
