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
	timer->stored_duration = -1.0;
	return timer;
}

Timer* timer_new(void) {
	Timer* timer = (Timer*)malloc(sizeof(Timer));
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

void timer_store(Timer* timer) {
	if (!timer) return;
	timer->stored_duration = timer->end - timer->start;
}

void timer_print(const Timer* timer, const char* name) {
	return timer_print_var(timer, name, false);
}

void timer_print_cmp(const Timer* timer, const char* name) {
	return timer_print_var(timer, name, true);
}

void timer_print_var(const Timer* timer, const char* name, bool do_cmp) {
	if (!timer || !name) return;
	const double dur = timer->end - timer->start;
	printf("%s time: %0.04fs", name, dur);
	if (do_cmp) {
		if (timer->stored_duration <= 0) {
			printf(" (%%--)");
		} else {
			const double percent = 100.0f * dur / timer->stored_duration;
			printf(" (%%%.2f)", percent);
		}
	}
	printf(".\n");
}