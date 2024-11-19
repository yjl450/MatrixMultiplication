
#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{

   // set your block dimensions and grid dimensions here
   gridDim.x = n / (TILESCALE_M*blockDim.x);
   gridDim.y = n / (TILESCALE_N*blockDim.x);

   // you can overwrite blockDim here if you like.
   if (n % (TILESCALE_M*blockDim.x) != 0)
      gridDim.x++;
   if (n % (TILESCALE_N*blockDim.x) != 0)
      gridDim.y++;
}
