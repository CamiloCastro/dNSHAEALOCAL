//
// Created by Juan on 29/08/2017.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "include/pcg_variants.h"
#include "include/entropy.h"

void main()
{
    srandom((unsigned int) time(NULL));
    pcg32_random_t rng;
    uint64_t seeds[2];
    seeds[0] = (uint64_t) random();
    seeds[1] = (uint64_t) random();
//    entropy_getbytes((void*)seeds, sizeof(seeds));
    pcg32_srandom_r(&rng, seeds[0], seeds[1]);
    printf("  Coins: ");

    for (int i = 0; i < 65; ++i)
        printf("%c", pcg32_boundedrand_r(&rng, 2) ? 'H' : 'T');
    printf("\n");

}
