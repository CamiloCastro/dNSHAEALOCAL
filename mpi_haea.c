#include "mpi.h"
#include "include/pcg_variants.h"
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#define ITERATIONS 500
#define POPSIZE 3072
#define OPERATORSNUMBER 2
#define DIMENSIONS 1001

void printBest(int* pop)
{
	int best = 0, i, pos_best = 0;
	for (i = 0; i < POPSIZE; i++)
	{
		if (pop[(i + 1)*DIMENSIONS - 1] > best)
		{
			best = pop[(i + 1)*DIMENSIONS - 1];
			pos_best = i;
		}
	}	
	int m;
	for (m = pos_best*DIMENSIONS; m < pos_best*DIMENSIONS + DIMENSIONS - 1; m++)
	{
		printf("%i", pop[m]);
	}
	printf(" %i\n", pop[m]);
}

double generate_random(pcg32_random_t *rng) {
    int h = pcg32_boundedrand_r(rng, 1000);
    return h / 1000.0;
}

void one_point_cross(int *newIndividuals, int *pop, int start, int end, pcg32_random_t *rng)
{
	int pos = pcg32_boundedrand_r(rng, DIMENSIONS - 1);
	int partner = pcg32_boundedrand_r(rng, POPSIZE);
	int cont = 0, sum = 0;
	for (int i = start; i < start + pos; i++)
	{
		newIndividuals[cont] = pop[i];
		sum += pop[i];
		cont++;
	}
	for (int i = DIMENSIONS*partner + pos; i < DIMENSIONS*partner + DIMENSIONS - 1; i++)
	{
		newIndividuals[cont] = pop[i];
		sum += pop[i];
		cont++;
	}
	newIndividuals[cont] = sum;
	cont++;
	sum = 0;
	for (int i = DIMENSIONS*partner; i < DIMENSIONS*partner + pos; i++)
	{
		newIndividuals[cont] = pop[i];
		sum += pop[i];
		cont++;
	}
	for (int i = start + pos; i < end - 1; i++)
	{
		newIndividuals[cont] = pop[i];
		sum += pop[i];
		cont++;
	}
	newIndividuals[cont] = sum;
}

void mutation(int *newIndividuals, int *pop, int start, int end, pcg32_random_t *rng)
{
	int pos = pcg32_boundedrand_r(rng, DIMENSIONS - 1);
	int cont = 0, sum = 0;
	for (int i = start; i < end - 1; i++) {
		if (i == start + pos)
			newIndividuals[cont] = 1 - pop[i];
		else
			newIndividuals[cont] = pop[i];

		sum += newIndividuals[cont];
		cont++;
	}
	newIndividuals[cont] = sum;

}

void applyOperators(double *operators_probabilites, int *pop, pcg32_random_t *rng, int row)
{	
	int i;
	int start, end;
	//Select Operator
	double sum = 0.0;
	int cont = 0;
	double rand_number = generate_random(rng);		
	for (int i = row * OPERATORSNUMBER; i < row * OPERATORSNUMBER + OPERATORSNUMBER; i++)
	{
		sum += operators_probabilites[i];
		if (rand_number < sum)
			break;
		cont++;
	}

	start = row * DIMENSIONS;
	end = start + DIMENSIONS;
	double reward = -1.0;
	//Cross
	int newIndividuals[2 * DIMENSIONS];
	if (cont == 0)
	{
		one_point_cross(newIndividuals, pop, start, end, rng);
		int cont_new;
		if (newIndividuals[DIMENSIONS - 1] > pop[end - 1] && newIndividuals[DIMENSIONS - 1] > newIndividuals[2 * DIMENSIONS - 1])
		{
			cont_new = 0;
			for (int i = start; i < end; i++)
			{
				pop[i] = newIndividuals[cont_new];
				cont_new++;
			}
			reward = 1.0;
		}
		else if (newIndividuals[2 * DIMENSIONS - 1] > pop[end - 1] && newIndividuals[2 * DIMENSIONS - 1] > newIndividuals[DIMENSIONS - 1])
		{
			cont_new = DIMENSIONS;
			for (int i = start; i < end; i++)
			{
				pop[i] = newIndividuals[cont_new];
				cont_new++;
			}
			reward = 1.0;
		}

	}
	//Mutation
	else if (cont == 1)
	{
		mutation(newIndividuals, pop, start, end, rng);
		if (newIndividuals[DIMENSIONS - 1] > pop[end - 1])
		{
			int cont_new = 0;
			for (int i = start; i < end; i++)
			{
				pop[i] = newIndividuals[cont_new];
				cont_new++;
			}
			reward = 1.0;
		}
	}

	//Apply reward

	double plus = generate_random(rng);
	plus = 1.0 + (plus * reward);
	operators_probabilites[row * OPERATORSNUMBER + cont] *= plus;

	//Normalizes
	float sumP = 0.0;
	for (int i = row * OPERATORSNUMBER; i < row * OPERATORSNUMBER + OPERATORSNUMBER; i++)
	{
		sumP += operators_probabilites[i];
	}

	for (int i = row * OPERATORSNUMBER; i < row * OPERATORSNUMBER + OPERATORSNUMBER; i++)
	{
		operators_probabilites[i] /= sumP;
	}
}






int main (int argc, char *argv[]) 
{
	// total nuber of processes	
	int total_proc;	
	
	// rank of each process
	int rank;

	int *pop,*pop_partial;
	
	double *operators_probability, *prob_partial;
	
	// elements per process	
	long long int n_per_proc;	
	
	long long int i, j, row;
	
	double start, end;
    
	
	// Initialization of MPI environment
	MPI_Init (&argc, &argv);
	
	//Now you know the total number of processes running in parallel
	MPI_Comm_size (MPI_COMM_WORLD, &total_proc);
	
	//Rank of actual process
	MPI_Comm_rank (MPI_COMM_WORLD,&rank);
	
	n_per_proc = POPSIZE/total_proc;
	int cont;
	
	pop_partial = (int *) malloc(sizeof(int)*DIMENSIONS*n_per_proc);
	prob_partial = (double *) malloc(sizeof(double)*OPERATORSNUMBER*n_per_proc);
	
	pop =  (int *) malloc(sizeof(int)*DIMENSIONS*POPSIZE);
	operators_probability = (double *) malloc(sizeof(double)*OPERATORSNUMBER*POPSIZE);
	
	pcg32_random_t rng;
	srandom((unsigned int) (time(NULL) + rank));
    uint64_t seeds[2];
    seeds[0] = (uint64_t) random();
    seeds[1] = (uint64_t) random();
    pcg32_srandom_r(&rng, seeds[0], seeds[1]);
	start = MPI_Wtime();
	
	if (rank == 0)
	{	
		for(i=0;i<n_per_proc;i++)
		{
			cont = 0;
			for(j=0;j<DIMENSIONS - 1;j++)
			{
				int num = pcg32_boundedrand_r(&rng, 2);
				pop_partial[i*DIMENSIONS + j] = num;
				cont += num;
			}
			pop_partial[i*DIMENSIONS + j] = cont;
			
			for(j=0;j<OPERATORSNUMBER;j++)
			{
				prob_partial[i*OPERATORSNUMBER + j] = 1.0/OPERATORSNUMBER;
			}
		}
		
		MPI_Gather(pop_partial, n_per_proc * DIMENSIONS, MPI_INT, pop, n_per_proc * DIMENSIONS, MPI_INT, 0, MPI_COMM_WORLD);	
		MPI_Gather(prob_partial, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, operators_probability, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, 0, MPI_COMM_WORLD);					
		
		for(int iter=0;iter<ITERATIONS;iter++)
		{
			MPI_Bcast (pop, POPSIZE*DIMENSIONS, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast (operators_probability, POPSIZE*OPERATORSNUMBER, MPI_DOUBLE, 0, MPI_COMM_WORLD);					
			int pos_pop = 0;
			int pos_prob = 0;
			for(row=0;row<n_per_proc;row++)
			{
				applyOperators(operators_probability, pop, &rng, row);	
				for (int k = row * OPERATORSNUMBER; k < row * OPERATORSNUMBER + OPERATORSNUMBER; k++)
				{
					prob_partial[pos_prob] = operators_probability[k];
					pos_prob++;
				}
				
				for (int k = row * DIMENSIONS; k < row * DIMENSIONS + DIMENSIONS; k++)
				{
					pop_partial[pos_pop] = pop[k];
					pos_pop++;
				}	
				
			}					
		
			MPI_Gather(pop_partial, n_per_proc * DIMENSIONS, MPI_INT, pop, n_per_proc * DIMENSIONS, MPI_INT, 0, MPI_COMM_WORLD);	
			MPI_Gather(prob_partial, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, operators_probability, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, 0, MPI_COMM_WORLD);										
		}
		end = MPI_Wtime();
		printBest(pop);
		printf("\nTime elapsed: %.4f\n", end - start);
		
	}
	else
	{	
		for(i=0;i<n_per_proc;i++)
		{
			cont = 0;
			for(j=0;j<DIMENSIONS - 1;j++)
			{
				int num = pcg32_boundedrand_r(&rng, 2);
				pop_partial[i*DIMENSIONS + j] = num;
				cont += num;
			}
			pop_partial[i*DIMENSIONS + j] = cont;
			
			for(j=0;j<OPERATORSNUMBER;j++)
			{
				prob_partial[i*OPERATORSNUMBER + j] = 1.0/OPERATORSNUMBER;
			}
		}
		
		MPI_Gather(pop_partial, n_per_proc * DIMENSIONS, MPI_INT, pop, n_per_proc * DIMENSIONS, MPI_INT, 0, MPI_COMM_WORLD);	
		MPI_Gather(prob_partial, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, operators_probability, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, 0, MPI_COMM_WORLD);		
		
		for(int iter=0;iter<ITERATIONS;iter++)
		{		
			MPI_Bcast (pop, POPSIZE*DIMENSIONS, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast (operators_probability, POPSIZE*OPERATORSNUMBER, MPI_DOUBLE, 0, MPI_COMM_WORLD);						
			
			int pos_pop = 0;
			int pos_prob = 0;
			for(row=rank * n_per_proc;row<rank * n_per_proc + n_per_proc;row++)
			{
				applyOperators(operators_probability, pop, &rng, row);
				for (int k = row * OPERATORSNUMBER; k < row * OPERATORSNUMBER + OPERATORSNUMBER; k++)
				{
					prob_partial[pos_prob] = operators_probability[k];
					pos_prob++;
				}
				
				for (int k = row * DIMENSIONS; k < row * DIMENSIONS + DIMENSIONS; k++)
				{
					pop_partial[pos_pop] = pop[k];
					pos_pop++;
				}
				
			}
			MPI_Gather(pop_partial, n_per_proc * DIMENSIONS, MPI_INT, pop, n_per_proc * DIMENSIONS, MPI_INT, 0, MPI_COMM_WORLD);	
			MPI_Gather(prob_partial, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, operators_probability, n_per_proc * OPERATORSNUMBER, MPI_DOUBLE, 0, MPI_COMM_WORLD);						
		}
	}
	
	

	MPI_Finalize();
}
