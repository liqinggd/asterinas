#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

unsigned long long rdtsc(void)
{
	unsigned long long lo, hi;
	asm volatile (	"rdtsc\n"
			: "=a"(lo), "=d"(hi)
			);
	return (unsigned long long) ( (hi<<32) | (lo) );
}

#define BUFFER_SIZE 8192 // 8KB
#define NUM_OF_CALLS 1000000

int main(int argc, char *argv[])
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <file_name>\n", argv[0]);
		return 1;
	}

	int fd = open(argv[1], O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "Failed to open file: %s\n", argv[1]);
		return 1;
	}

	char buffer[BUFFER_SIZE];
	ssize_t bytes_read;
	unsigned long long start, end;
	unsigned long long total_cycles, avg_latency;

	memset(buffer, 0, BUFFER_SIZE);

	start = rdtsc();
	for (int i = 0; i < NUM_OF_CALLS; i++) {
		bytes_read = read(fd, buffer, BUFFER_SIZE);
		if (bytes_read == 0) {
			if (lseek(fd, 0, SEEK_SET) < 0) {
				fprintf(stderr, "Failed to lseek");
				return 1;
			}
			continue;
		} else if (bytes_read == -1) {
			fprintf(stderr, "Failed to read");
			return 1;
		}
	}
	end = rdtsc();

	total_cycles = end - start;
	avg_latency = total_cycles / NUM_OF_CALLS;

	printf("Executed the read() (buffer size %d) syscall %d times.\n",
	       BUFFER_SIZE, NUM_OF_CALLS);
	printf("Syscall average latency: %lld cycles.\n", avg_latency);

	close(fd);
	return 0;
}