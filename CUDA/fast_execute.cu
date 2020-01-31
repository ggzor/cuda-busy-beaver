#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <iostream>

#define ITERATIONS 104000
#define BATCH_SIZE 2097152
#define MAX_REGISTER 5
#define SEED 314159

#define MAX_ZEROES 3
#define MAX_TRANSFERS 7
#define MAX_JUMPS 4

#define MAX_INSTRUCTIONS 150
#define PROGRAM_LINES 10

#define PROGRAM_SIZE (40 * sizeof(unsigned char))
#define BLOCK_SIZE 256

#define TOTAL_BLOCKS (BATCH_SIZE / BLOCK_SIZE)
#define TOTAL_PROGRAMS_MEMORY (PROGRAM_SIZE * BATCH_SIZE * sizeof(unsigned char))
#define TOTAL_RESULTS_MEMORY (BATCH_SIZE * sizeof(unsigned char))
#define TOTAL_RANDOM_STATE_SIZE (BATCH_SIZE * sizeof(curandState))

__global__
void initialize_states(curandState *states) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  curand_init(SEED, index, 0, &states[index]);
}

#define random(min, max) ((unsigned char)truncf(curand_uniform(&state) * (max - min + 0.999999f) + min))

__global__
void compute_program(int n, int it, unsigned char *programs, unsigned char *results, curandState *states, unsigned short *executedInstructions) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // Get local copy
  unsigned char P[PROGRAM_SIZE];
  curandState state = states[index];
  memcpy(P, &programs[PROGRAM_SIZE * index], PROGRAM_SIZE);

  // Generate program
  int zeroes = MAX_ZEROES;
  int jumps = MAX_JUMPS;
  int transfers = MAX_TRANSFERS;

  for (int i = 0; i < PROGRAM_LINES; i++) {
    if (i + 1 == PROGRAM_LINES)
      zeroes = 0;

    if (zeroes == 0) {
      if (jumps == 0) {
        if (transfers == 0) {
          P[i * 4] = 0;
        } else {
          P[i * 4] = random(1, 2);
        }
      } else {
        if (transfers == 0) {
          P[i * 4] = 1 + random(0, 1) * 2;
        } else {
          P[i * 4] = random(1, 3);
        }
      }
    } else {
      if (jumps == 0) {
        if (transfers == 0) {
          P[i * 4] = random(0, 1);
        } else {
          P[i * 4] = random(0, 2);
        }
      } else {
        if (transfers == 0) {
          P[i * 4] = random(0, 2);

          // Map 2 -> 3
          if (P[i * 4] == 2)
            P[i * 4]++;
        } else {
          P[i * 4] = random(0, 3);
        }
      }
    }

    if (P[i * 4] == 0) {
      zeroes--;
      P[i * 4 + 1] = random(1, MAX_REGISTER);
    } else if (P[i * 4] == 1) {
      P[i * 4 + 1] = random(1, MAX_REGISTER);
    } else if (P[i * 4] == 2) {
      transfers--;
      P[i * 4 + 1] = random(1, MAX_REGISTER);
      P[i * 4 + 2] = random(2, MAX_REGISTER);

      if (P[i * 4 + 1] == P[i * 4 + 2]) {
        P[i * 4 + 2] = 1;
      }      
    } else if (P[i * 4] == 3) {
      jumps--;

      P[i * 4 + 1] = random(1, MAX_REGISTER);
      P[i * 4 + 2] = random(1, MAX_REGISTER);
      P[i * 4 + 3] = random(1, PROGRAM_LINES + 1);
    }
  }

  // Execution
  unsigned short count = 0;
  int ip = 0;
  int R[MAX_REGISTER];

  for (int i = 0; i < MAX_REGISTER; i++)
    R[i] = 0;

  while ((0 <= ip && ip < PROGRAM_LINES) && count < MAX_INSTRUCTIONS) {
    count++;

    int kind = P[ip * 4 + 0];
    int p1 = P[ip * 4 + 1];
    int p2 = P[ip * 4 + 2];
    int p3 = P[ip * 4 + 3];

    if (kind == 0)
      R[p1 - 1] = 0;
    else if (kind == 1)
      R[p1 - 1]++;
    else if (kind == 2)
      R[p2 - 1] = R[p1 - 1];
    else if (kind == 3 && (R[p1 - 1] == R[p2 - 1])) {
      ip = p3 - 1;
      continue;
    }
      
    ip += 1;
  }

  if (count < MAX_INSTRUCTIONS) {
    results[index] = R[0];
    executedInstructions[index] = count;
  }

  // Reload memory
  states[index] = state;
  memcpy(&programs[PROGRAM_SIZE * index], P, PROGRAM_SIZE);
}

void print_program(unsigned char *program) {
  for (int i = 0; i < PROGRAM_LINES; i++) {
    int kind = program[i * 4 + 0];
    int p1 = program[i * 4 + 1];
    int p2 = program[i * 4 + 2];
    int p3 = program[i * 4 + 3];

    if (kind == 0)
      printf("Z(%d)\n", p1);
    else if (kind == 1)
      printf("S(%d)\n", p1);
    else if (kind == 2)
      printf("T(%d,%d)\n", p1, p2);
    else if (kind == 3)
      printf("J(%d,%d,%d)\n", p1, p2, p3);
  }
}

int main() {
  unsigned char bestProgram[PROGRAM_SIZE];
  unsigned char bestProgramResult = 0;
  unsigned short bestExecutedInstructions = 1000;

  curandState *randomStates;
  unsigned char *programs, *results;
  unsigned short *executedInstructions;

  cudaMalloc(&randomStates, TOTAL_RANDOM_STATE_SIZE);
  cudaMallocManaged(&programs, TOTAL_PROGRAMS_MEMORY);
  cudaMallocManaged(&results, TOTAL_RESULTS_MEMORY);
  cudaMallocManaged(&executedInstructions, BATCH_SIZE * sizeof(unsigned short));

  int device;
  cudaGetDevice(&device);

  initialize_states<<<TOTAL_BLOCKS, BLOCK_SIZE>>>(randomStates);

  for (int i = 0; i < ITERATIONS; i++) {
    cudaMemPrefetchAsync(programs, TOTAL_PROGRAMS_MEMORY, device, NULL);
    cudaMemPrefetchAsync(results, TOTAL_RESULTS_MEMORY, device, NULL);
    cudaMemPrefetchAsync(executedInstructions, BATCH_SIZE * sizeof(unsigned short), device, NULL);

    compute_program<<<TOTAL_BLOCKS, BLOCK_SIZE>>>(BATCH_SIZE, i, programs, results, randomStates, executedInstructions);
    cudaDeviceSynchronize();

    for (int j = 0; j < BATCH_SIZE; j++) {
      if (results[j] > bestProgramResult || (results[j] == bestProgramResult && executedInstructions[j] < bestExecutedInstructions)) {
        cudaMemcpy(bestProgram, &programs[j * PROGRAM_SIZE], PROGRAM_SIZE, cudaMemcpyDeviceToHost);
        bestProgramResult = results[j];
        bestExecutedInstructions = executedInstructions[j];

        printf("Better program found: %2d in %3d instructions (@%d[%d])\n", bestProgramResult, bestExecutedInstructions, i, j);
        print_program(bestProgram);
      }

    }
  }

  cudaFree(randomStates);
  cudaFree(programs);
  cudaFree(results);
}