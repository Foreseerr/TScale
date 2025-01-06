/******************************************************************************
*  Module:  r250.c   
*  Description: implements R250 random number generator, from S. 
*  Kirkpatrick and E.  Stoll, Journal of Computational Physics, 40, p. 
*  517 (1981).
*  Written by:    W. L. Maier
*
*	METHOD...
*		16 parallel copies of a linear shift register with
*		period 2^250 - 1.  FAR longer period than the usual
*		linear congruent generator, and commonly faster as
*		well.  (For details see the above paper, and the
*		article in DDJ referenced below.)
*
*	HISTORY...
*		Sep 92: Number returned by dr250() is in range [0,1) instead 
*			of [0,1], so for example a random angle in the
*			interval [0, 2*PI) can be calculated
*			conveniently.  (J. R. Van Zandt <jrv@mbunix.mitre.org>)
*		Aug 92: Initialization is optional.  Default condition is 
*			equivalent to initializing with the seed 12345,
*			so that the sequence of random numbers begins:
*			1173, 53403, 52352, 35341...  (9996 numbers
*			skipped) ...57769, 14511, 46930, 11942, 7978,
*			56163, 46506, 45768, 21162, 43113...  Using ^=
*			operator rather than two separate statements. 
*			Initializing with own linear congruent
*			pseudorandom number generator for portability. 
*			Function prototypes moved to a header file. 
*			Implemented r250n() to generate numbers
*			uniformly distributed in a specific range
*			[0,n), because the common expedient rand()%n is
*			incorrect.  (J. R. Van Zandt <jrv@mbunix.mitre.org>)
*		May 91: Published by W. L. Maier, "A Fast Pseudo Random Number 
*			Generator", Dr. Dobb's Journal #176.
******************************************************************************/

#include "stdafx.h"
#include <stdlib.h>

/**** Static variables ****/
static int r250_index = 0;
static unsigned int r250_buffer[250] = {
	15301,57764,10921,56345,19316,43154,54727,49252,32360,49582,
	26124,25833,34404,11030,26232,13965,16051,63635,55860,5184,
	15931,39782,16845,11371,38624,10328,9139,1684,48668,59388,
	13297,1364,56028,15687,63279,27771,5277,44628,31973,46977,
	16327,23408,36065,52272,33610,61549,58364,3472,21367,56357,
	56345,54035,7712,55884,39774,10241,50164,47995,1718,46887,
	47892,6010,29575,54972,30458,21966,54449,10387,4492,644,
	57031,41607,61820,54588,40849,54052,59875,43128,50370,44691,
	286,12071,3574,61384,15592,45677,9711,23022,35256,45493,
	48913,146,9053,5881,36635,43280,53464,8529,34344,64955,
	38266,12730,101,16208,12607,58921,22036,8221,31337,11984,
	20290,26734,19552,48,31940,43448,34762,53344,60664,12809,
	57318,17436,44730,19375,30,17425,14117,5416,23853,55783,
	57995,32074,26526,2192,11447,11,53446,35152,64610,64883,
	26899,25357,7667,3577,39414,51161,4,58427,57342,58557,
	53233,1066,29237,36808,19370,17493,37568,3,61468,38876,
	17586,64937,21716,56472,58160,44955,55221,63880,1,32200,
	62066,22911,24090,10438,40783,36364,14999,2489,43284,9898,
	39612,9245,593,34857,41054,30162,65497,53340,27209,45417,
	37497,4612,58397,52910,56313,62716,22377,40310,15190,34471,
	64005,18090,11326,50839,62901,59284,5580,15231,9467,13161,
	58500,7259,317,50968,2962,23006,32280,6994,18751,5148,
	52739,49370,51892,18552,52264,54031,2804,17360,1919,19639,
	2323,9448,43821,11022,45500,31509,49180,35598,38883,19754,
	987,11521,55494,38056,20664,2629,50986,31009,54043,59743
	};

static unsigned myrand();
static void mysrand(unsigned newseed);

/**** Function: r250_init  
	Description: initializes r250 random number generator. ****/
void r250_init(int seed)
{
/*---------------------------------------------------------------------------*/
    int        j, k;
    unsigned int mask;
    unsigned int msb;
/*---------------------------------------------------------------------------*/
    mysrand(seed);
    r250_index = 0;
    for (j = 0; j < 250; j++)     /* Fill the r250 buffer with 15-bit values */
        r250_buffer[j] = myrand();
    for (j = 0; j < 250; j++)     /* Set some of the MS bits to 1 */
        if (myrand() > 16384)
            r250_buffer[j] |= 0x8000;
    msb = 0x8000;       /* To turn on the diagonal bit   */
    mask = 0xffff;      /* To turn off the leftmost bits */
    for (j = 0; j < 16; j++)
        {
        k = 11 * j + 3;             /* Select a word to operate on        */
        r250_buffer[k] &= mask;     /* Turn off bits left of the diagonal */
        r250_buffer[k] |= msb;      /* Turn on the diagonal bit           */
        mask >>= 1;
        msb >>= 1;
        }
}

static struct SInitRandom250
{
	SInitRandom250()
	{
		r250_init( 13 );
	}
} doit;
/**** Function: r250 Description: returns a random unsigned integer k
				uniformly distributed in the interval 0 <= k < 65536.  ****/
unsigned int r250()
{
/*---------------------------------------------------------------------------*/
    int    j;
    unsigned int new_rand;
/*---------------------------------------------------------------------------*/
    if (r250_index >= 147)
        j = r250_index - 147;      /* Wrap pointer around */
    else
        j = r250_index + 103;

    new_rand = r250_buffer[r250_index] ^= r250_buffer[j];

    if (r250_index >= 249)      /* Increment pointer for next time */
        r250_index = 0;
    else
        r250_index++;

    return new_rand;
}

/**** Function: r250n Description: returns a random unsigned integer k
					uniformly distributed in the interval 0 <= k < n ****/
unsigned int r250n(unsigned n)
{
	unsigned int new_rand;
	unsigned int limit = (0xffffffffU / n) * n;
	do 
	{
		new_rand = (r250() << 16) + r250();
	} while(new_rand >= limit);
	return new_rand % n;
 //   register int    j;
 //   register unsigned int new_rand, limit;
	//limit = (65535U/n)*n;
	//do 
 //       {//new_rand = r250();
 //       if (r250_index >= 147)
 //           j = r250_index - 147;      /* Wrap pointer around */
 //       else
 //           j = r250_index + 103;

 //       new_rand = r250_buffer[r250_index] ^= r250_buffer[j];

 //       if (r250_index >= 249)      /* Increment pointer for next time */
 //           r250_index = 0;
 //       else
 //           r250_index++;
 //       } while(new_rand >= limit);
 //   return new_rand%n;
}

/**** Function: dr250 
		Description: returns a random double z in range 0 <= z < 1.  ****/
double dr250()
{
/*---------------------------------------------------------------------------*/
    int    j;
    unsigned int new_rand;
/*---------------------------------------------------------------------------*/
    if (r250_index >= 147)
        j = r250_index - 147;     /* Wrap pointer around */
    else
        j = r250_index + 103;

    new_rand = r250_buffer[r250_index] ^= r250_buffer[j];

    if (r250_index >= 249)      /* Increment pointer for next time */
        r250_index = 0;
    else
        r250_index++;
    return new_rand / 65536.;   /* Return a number in [0.0 to 1.0) */
}


/***  linear congruent pseudorandom number generator for initialization ***/

static unsigned long seed=1;

	/*	Return a pseudorandom number in the interval 0 <= n < 32768.
		This produces the following sequence of pseudorandom 
		numbers:
		346, 130, 10982, 1090...  (9996 numbers skipped) ...23369,
		2020, 5703, 12762, 10828, 16252, 28648, 27041, 23444, 6604... 
	*/ 
static unsigned myrand()
{
	seed = seed*0x015a4e35L + 1;
	return (seed>>16)&0x7fff;
}

	/*	Initialize above linear congruent pseudorandom number generator */
static void mysrand(unsigned newseed)
{	seed = newseed;
}

