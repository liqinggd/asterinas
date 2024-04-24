// SPDX-License-Identifier: MPL-2.0

#define _GNU_SOURCE
#include <stdio.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

unsigned long long rdtsc(void)
{
	unsigned long long lo, hi;
	asm volatile (	"rdtsc\n"
			: "=a"(lo), "=d"(hi)
			);
	return (unsigned long long) ( (hi<<32) | (lo) );
}

#define NUM_OF_CALLS 1000000

int main()
{
	unsigned long long start, end;
	unsigned long long total_cycles, avg_latency;
	pid_t pid;

	start = rdtsc();

	for (int i = 0; i < NUM_OF_CALLS; i++) {
		pid = syscall(SYS_getpid);
	}

	end = rdtsc();

	total_cycles = end - start;
	avg_latency = total_cycles / NUM_OF_CALLS;

	printf("Process %d executed the getpid() syscall %d times.\n", pid,
	       NUM_OF_CALLS);
	printf("Syscall average latency: %lld cycles.\n", avg_latency);

	return 0;
}
