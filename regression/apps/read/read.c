#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

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
	struct timespec start, end;
	long long seconds, nanoseconds, total_nanoseconds, avg_latency;

	clock_gettime(CLOCK_MONOTONIC, &start);

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

	clock_gettime(CLOCK_MONOTONIC, &end);

	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;

	total_nanoseconds = seconds * 1e9 + nanoseconds;
	avg_latency = total_nanoseconds / NUM_OF_CALLS;

	printf("Executed the read() (buffer size %d) syscall %d times.\n",
	       BUFFER_SIZE, NUM_OF_CALLS);
	printf("Syscall average latency: %lld nanoseconds.\n", avg_latency);

	close(fd);
	return 0;
}