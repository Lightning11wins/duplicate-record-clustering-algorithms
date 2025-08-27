#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "timer.h"

double monotonic_seconds(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9f;
}

Timer* timer_init(Timer* timer) {
	if (!timer) return timer;
	timer->start = -1.0;
	timer->end = -1.0;
	timer->stored_duration = NAN;
	return timer;
}

Timer* timer_new(void) {
	Timer* timer = malloc(sizeof(Timer));
	timer_init(timer);
	return timer;
}

void timer_start(Timer* timer) {
	if (!timer) return;
	timer->start = monotonic_seconds();
}

void timer_stop(Timer* timer) {
	if (!timer) return;
	timer->end = monotonic_seconds();
}

double timer_get(Timer* timer) {
	return (timer) ? timer->end - timer->start : NAN;
}

double timer_store(Timer* timer) {
	if (!timer) return NAN;
	return timer->stored_duration = timer_get(timer);
}

void timer_print(const Timer* timer, const char* name) {
	return timer_print_cmp(timer, name, NAN);
}

void timer_print_s(const Timer* timer, const char* name) {
	return timer_print_cmp(timer, name, timer->stored_duration);
}

void timer_print_cmp(const Timer* timer, const char* name, double cmp) {
	if (!timer || !name) return;
	const double dur = timer->end - timer->start;
	printf("%s time: %.4fs", name, dur);
	if (!isnan(cmp)) {
		if (cmp <= 0) {
			printf(" (%%--)");
		} else {
			const double percent = 100.0f * dur / cmp;
			printf(" (%%%.2f)", percent);
		}
	}
	printf(".\n");
}

void timer_free(Timer* timer) {
	free(timer);
}
