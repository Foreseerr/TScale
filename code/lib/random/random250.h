#pragma once

/**** Function: r250 Description: returns a random unsigned integer k
uniformly distributed in the interval 0 <= k < 65536.  ****/
unsigned int r250();

/**** Function: r250n Description: returns a random unsigned integer k
uniformly distributed in the interval 0 <= k < n ****/
unsigned int r250n(unsigned n);

void r250_init(int seed);
