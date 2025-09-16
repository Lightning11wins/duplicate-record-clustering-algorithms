#include <assert.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "utils.h"

// 1M: 185.66 MB
void fprint_mem(FILE* out) {
	FILE *fp = fopen("/proc/self/statm", "r");
	if (!fp) { perror("fopen"); return; }

	long size, resident, share, text, lib, data, dt;
	if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
			&size, &resident, &share, &text, &lib, &data, &dt) != 7) {
		fprintf(stderr, "Failed to read memory info\n");
		fclose(fp);
		return;
	}
	fclose(fp);

	long page_size = sysconf(_SC_PAGESIZE); // in bytes
	long resident_bytes = resident * page_size;

	fprintf(out, INDENT"Memory used (RSS): %ld bytes (%.2f MB)\n", resident_bytes, (double)resident_bytes / (1024.0 * 1024.0));
	fprintf(out, INDENT"Share %ldb, Text %ldb, Lib %ldb, Data %ldb\n", share, text, lib, data);
}

// Stacktrace that lies to you. Idk why.
void fprint_stacktrace(void) {
	// Define an array to hold the return addresses.
	void *callstack[64];
	int frames = backtrace(callstack, 64);
	char** symbol_list = backtrace_symbols(callstack, frames);
	if (symbol_list == NULL) { return perror("backtrace_symbols failed"); }

	// Print the backtrace.
	fprintf(stderr, "Stacktrace (%d):\n", frames);
	repeat (frames, i) {
		char* symbol = symbol_list[i];
		
		// Attempt to get print line number.
		unsigned int line_number = 0;
		if (strneql(symbol, "./bin/kmeans_debug.(", 20u)) goto fail;
		char* start = strchr(symbol, '+');
		if (start == NULL) goto fail;
		if (sscanf(++start, "%x", &line_number) != 1) goto fail;
		char fname[BUFSIZ];
		snprintf(memset(fname, 0, sizeof(fname)), max(sizeof(fname), (unsigned)((unsigned long)start - (unsigned long)symbol - 20u)), "%s", symbol + 20);
		fprintf(stderr, "./kmeans.c#%s:%u\n", fname, line_number);
		continue;
		
		fail:
		fprintf(stderr, "%s\n", symbol);
	}
	free(symbol_list);
}

// Function for failing on error.
void fail(const char* function_name) {
	// Create the most descriptive error message we can.
	char error_buf[BUFSIZ];
	snprintf(error_buf, sizeof(error_buf), "kmeans.c: Fail - %s", function_name);
	perror(error_buf);
	
	// Throw error for easier locating in a debugger.
	fprintf(stderr, "Program will now crash.\n");
	assert(0);
}

/*** Helper function for compact error handling on library & system function calls.
 *** Any non-zero value is treated as an error, exiting the program.
 ***
 *** @param result The result of the function we're checking.
 *** @param function_name The name of the function being checked (for debugging).
 *** @returns result
 ***/
int check(int result, const char* function_name) {
	if (result != 0) fail(function_name);
	return result;
}

/*** Helper function for compact error handling on library & system function calls.
 *** Any null value is treated as an error, exiting the program.
 ***
 *** @param result The result of the function we're checking.
 *** @param function_name The name of the function being checked (for debugging).
 *** @returns result
 ***/
void* check_ptr(void* result, const char* function_name) {
	if (result == NULL) fail(function_name);
	return result;
}

/*** Safe malloc with error handling.
 *** 
 *** @param size The size of memory to malloc.
 *** @returns Clean, allocated memory.
 ***/
void* mallocs(const size_t size) {
	void* ptr = malloc(size);
	if (ptr == NULL) {
		// Create the most descriptive error message we can.
		char error_buf[BUFSIZ];
		snprintf(error_buf, sizeof(error_buf), "kmeans.c: Fail - malloc(%lu bytes)", size);
		perror(error_buf);
		
		// Throw error for easier locating in a debugger.
		fprintf(stderr, "Program will now crash.\n");
		assert(0);
	}
	return memset(ptr, 0, size);
}