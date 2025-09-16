#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "lib/arraylist.h"
#include "lib/timer.h"
#include "lib/utils.h"

#define NUM_DIMS 251
#define SEED 1621963727
#define DUPE_THRESHOLD 0.75
#define KMEANS_IMPROVEMENT_THRESHOLD 0.0002

// Test Parameters
unsigned int window_sizes[] = {3, 6, 16, 32, 64, 256};
unsigned int cluster_counts[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
unsigned int dataset_sizes[] = {1000, 10000};//, 100000};
unsigned int max_iter = 64;
// #define PATH_FORMAT "datasets/dataset_updated.txt"
#define PATH_FORMAT "datasets/surnames_%u.txt"
// #define PATH_FORMAT "datasets/phone_numbers_%u.txt"

FILE* complete_file = NULL;
FILE* sliding_file = NULL;
FILE* kmeans_file = NULL;

/*** Complexity Analysis
 *** n: The number of input records.
 *** d: The number of dimensions. (251)
 *** w: The number of records in sliding window.
 *** k: The number of kmeans clusters.
 *** i: The number of kmeans iterations.
 *** s: The number of records in the average cluster.
 ***/
 
// ====================================
// Helper functions

void fprint_vector(FILE* out, const int* vector) {
	for (unsigned int j = 0, idx = 0; idx < NUM_DIMS; j++) {
		if (vector[j] > 0) { fprintf(out, " % 3d,", vector[j]); idx++; }
		else {
			unsigned int num = (unsigned)(-vector[j]);
			idx += num;
			repeat (num, k) fprintf(out, "    ,");
		}
	}
}

void fprint_centroid(FILE* out, const double* vector) {
	repeat (NUM_DIMS, i) fprintf(out, " % 3g,", vector[i]);
}

unsigned int get_char_pair_hash(unsigned int num1, unsigned int num2) {
	double sum = (num1 * num1 * num1) + (num2 * num2 * num2);
	double scale = ((double)num1 + 1.0) / ((double)num2 + 1.0);
	unsigned int hash = (unsigned int)round(sum * scale) - 1u;
	return hash % NUM_DIMS;
}

/*** Builds a vector using a string.
 *** 
 *** @param strs The string used to build the vector.
 *** @returns The built vector.
 ***/
int* build_vector(char* str) {
	// Malloc space for a dense vector.
	unsigned int* dense_vector = (unsigned int*)mallocs(NUM_DIMS * sizeof(unsigned int));
	
	// j is the former character, i is the latter.
	int num_chars = (int)strlen(str);
	for (int j = -1, i = 0; i <= (int)num_chars; i++) {
		// If latter character is punctuation or whitespace, skip it.
		if (ispunct(str[i]) || isspace(str[i])) continue;
		
		unsigned int temp1 = (j == -1) ? '`' : (unsigned int)tolower(str[j]);
		unsigned int temp2 = (i == num_chars) ? '`' : (unsigned int)tolower(str[i]);

		// If either character is a number, reassign the code point
		// The significance of 75 here seems to be that it puts the numbers
		// right after the end of the lowercase letters, but they will still
		// colide with the {, |, }, ~ and DEL ASCII characters.
		if ('0' <= temp1 && temp1 <= '9') temp1 += 75u;
		if ('0' <= temp2 && temp2 <= '9') temp2 += 75u;
		
		// Hash the character pair into an index (dimension).
		unsigned int dim = get_char_pair_hash(temp1, temp2);
		
		// Increment the dimension of the dense vector by a number from 1 to 13.
		dense_vector[dim] += (temp1 + temp2) % 13 + 1;
		
		j = i;
	}
	
	// Count how much space is needed for a sparse vector.
	bool zero_prev = false;
	unsigned int size = 0u;
	repeat (NUM_DIMS, j) {
		if (dense_vector[j] == 0u) {
			size += (zero_prev) ? 0u : 1u;
			zero_prev = true;
		} else {
			size++;
			zero_prev = false;
		}
	}

	// Convert the dense vector above to a sparse vector.
	int* sparse_vector = (int*)mallocs(size * sizeof(int));
	unsigned int j = 0u, sparse_idx = 0u;
	while (j < NUM_DIMS) {
		if (dense_vector[j] == 0u) {
			// Count and store consecutive zeros, except the first one,
			// which we already know is zero.
			unsigned int zero_count = 1;
			j++;
			while (j < NUM_DIMS && dense_vector[j] == 0u) {
				zero_count++;
				j++;
			}
			sparse_vector[sparse_idx++] = (int)-zero_count;
		} else {
			// Store the value
			sparse_vector[sparse_idx++] = (int)dense_vector[j++];
		}
	}
	
	// Free unused dense vector.
	free(dense_vector);
	
	return sparse_vector;
}

/*** Compute the value for `k` (number of clusters), given a dataset of with
 *** a size of `n`.
 *** 
 *** The following table shows data sizes vs.selected cluster size. In testing,
 *** these numbers tended to givea good balance of accuracy and dulocates detected.
 *** 
 *** ```csv
 *** Data Size, Actual
 *** 10k,       12
 *** 100k,      33
 *** 1M,        67
 *** ```
 *** 
 *** This function is not intended for datasets smaller than (`n < ~2000`).
 *** These should be handled using complete search.
 *** 
 *** LaTeX Notation: \log_{36}\left(n\right)^{3.1}-8
 *** 
 *** @param n The size of the dataset.
 *** @returns k, the number of clusters to use.
 ***/
unsigned int compute_k(unsigned int n) {
	return (unsigned)max(2, pow(log(n) / log(36), 3.2) - 8);
}

double magnitude_sparse(const int* vector) {
	unsigned int magnitude = 0;
	for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
		const int val = vector[i++];
		
		// Negative val represents -val 0s in the array, so skip that many values. 
		if (val < 0) dim += (unsigned)(-val);
		
		// We have a value, so square it and add it to the magnitude.
		else {magnitude += (unsigned)(val * val); dim++;}
	}
	return sqrt((double)magnitude);
}

double magnitude_dense(const double* vector) {
	double magnitude = 0.0;
	repeat (NUM_DIMS, i) magnitude += vector[i] * vector[i];
	return sqrt(magnitude);
}

void parse_token(const int token, unsigned int* remaining, unsigned int* value) {
	if (token < 0) {
		// This run contains -token zeros.
		*remaining = (unsigned)(-token);
		*value = 0u;
	} else {
		// This run contains one value.
		*remaining = 1;
		*value = (unsigned)(token);
	}
}

/*** Calculate similarity on sparce vectors.
 *** 
 *** @param v1 Sparse vector 1.
 *** @param v1 Sparse vector 2.
 *** @returns Similarity between 0 and 1 where
 *** 	1 indicates completely different and
 *** 	0 indicates identical.
 *** 
 *** Complexity: `O(3d)`
 ***/
double sparse_similarity(const int* v1, const int* v2) {
	// Calculate dot product.
	unsigned int val1 = 0, val2 = 0, dot_product = 0;
	unsigned int vec1_remaining = 0, vec2_remaining = 0;
	unsigned int dim = 0, i1 = 0, i2 = 0;
	while (dim < NUM_DIMS) {
		if (vec1_remaining == 0) parse_token(v1[i1++], &vec1_remaining, &val1);
		if (vec2_remaining == 0) parse_token(v2[i2++], &vec2_remaining, &val2);

		// Accumulate the dot_product. If either vector is 0 here,
		// the total is 0 and this statement does nothing.
		dot_product += val1 * val2;
		
		// Consume overlap from both runs.
		unsigned int overlap = min(vec1_remaining, vec2_remaining);
		vec1_remaining -= overlap;
		vec2_remaining -= overlap;
		dim += overlap;
	}
	
	// Optional optimization to speed up nonsimilar vectors.
	if (dot_product == 0) return 0.0;
	
	// Return the difference score.
	return (double)dot_product / (magnitude_sparse(v1) * magnitude_sparse(v2));
}
#define sparse_dif(v1, v2) (1.0 - sparse_similarity(v1, v2))

double sparse_similarity_c(const int* v1, const double* c2) {
	// Calculate dot product
	double dot_product = 0.0;
	for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
		const int val = v1[i++];
		
		// Negative val represents -val 0s in the array, so skip that many values. 
		if (val < 0) dim += (unsigned)(-val);
		
		// We have a value, so square it and add it to the magnitude.
		else dot_product += (double)val * c2[dim++];
	}
	
	// Return the difference score.
	return dot_product / (magnitude_sparse(v1) * magnitude_dense(c2));
}
#define sparse_dif_c(v1, c2) (1.0 - sparse_similarity_c(v1, c2))

/*** Calculate the average size of all clusters in a set of vectors.
 *** 
 *** @param vectors The vectors of the dataset.
 *** @param num_vectors The number of vectors in the dataset.
 *** @param labels The clusters to which vectors are assigned.
 *** @param centroids The locations of the centroids.
 *** @param num_clusters The number of centroids (k).
 *** @returns The average cluster size.
 ***/
double get_cluster_size(int** vectors, unsigned int num_vectors, unsigned int* labels, double** centroids, unsigned int num_clusters) {
	double cluster_sums[num_clusters];
	unsigned int cluster_counts[num_clusters];
	memset(cluster_sums, 0, sizeof(cluster_sums));
	memset(cluster_counts, 0, sizeof(cluster_counts));
	
	// Sum the difference from each vector to its cluster centroid.
	repeat (num_vectors, i) {
		unsigned int label = labels[i];
		cluster_sums[label] += sparse_dif_c(vectors[i], centroids[label]);
		cluster_counts[label]++;
	}
	
	// Add up the average cluster size.
	double cluster_total = 0.0;
	unsigned int num_valid_clusters = 0;
	repeat (num_clusters, label) {
		unsigned int cluster_count = cluster_counts[label];
		if (cluster_count == 0) continue;
		
		cluster_total += cluster_sums[label] / cluster_count;
		num_valid_clusters++;
	}
	
	// Return average sizes.
	return cluster_total / num_valid_clusters;
}

/*** Executes the k-means clustering algorithm. Selects NUM_CLUSTERS random
 *** vectors as initial centroids. Then points are assigned to the nearest
 *** centroid, after which centroids are moved to the center of their points.
 ***
 *** @param vectors The vectors to cluster.
 *** @param num_vectors The number of vectors to cluster.
 *** @param labels Stores the final cluster identities of the vectors after
 *** 	clustering is completed.
 *** @param centroids Stores the locations of the centroids used for the clusters
 *** 	of the data.
 *** @param iterations The number of iterations that actually executed is stored
 *** 	here. Leave this NULL if you don't care.
 *** @param max_iter The max number of iterations.
 *** @param num_clusters The number of clusters to generate.
 ***
 *** @attention - Assumes: NUM_CLUSTERS is in scope and is the number of centroids.
 *** @attention - Assumes: MAX_ITER is in scope and is the max number of iterations.
 ***
 *** @attention - Issue: At larger numbers of clustering iterations, some
 *** 	clusters have a size of	negative infinity. In this implementation,
 *** 	the bug is mitigated by setting a small number of max iterations,
 *** 	such as 16 instead of 100.
 *** @attention - Issue: Clusters do not apear to improve much after the first
 *** 	iteration, which puts the efficacy of the algorithm into question. This
 *** 	may be due to the uneven density of a typical dataset. However, the
 *** 	clusters still offer useful information.
 *** 
 *** Complexity:
 *** 
 *** - `O(kd + k + i*(k + n*(k+d) + kd))`
 *** 
 *** - `O(kd + k + ik + ink + ind + ikd)`
 *** 
 *** - `O(nk + nd)`
 ***/
void kmeans(int** vectors, unsigned int num_vectors, unsigned int* labels, double** centroids, unsigned int* iterations, unsigned int max_iter, unsigned int num_clusters) {
	// Select random vectors to use as the initial centroids.
	srand(SEED);
	repeat (num_clusters, i) {
		// Pick a random vector.
		const unsigned int random_index = (unsigned int)rand() % num_vectors;
		
		// Sparse copy the vector into a densely allocated centroid.
		double* centroid = centroids[i];
		const int* vector = vectors[random_index];
		for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
			int token = vector[i++];
			if (token > 0) centroid[dim++] = (double)token;
			else repeat (-token, j) centroid[dim++] = 0.0;
		}
	}
	
	// Allocate memory for new centroids.
	double** new_centroids = (double**)mallocs(num_clusters * sizeof(double*));
	repeat (num_clusters, i) {
		new_centroids[i] = (double*)mallocs(NUM_DIMS * sizeof(double));
	}
	
	// Main kmeans loop.
	double previous_average_cluster_size = 1.0;
	unsigned int* cluster_counts = (unsigned int*)mallocs(num_clusters * sizeof(unsigned int));
	repeat (max_iter, iter) {
		bool changed = false;
		
		// Reset new centroids.
		repeat (num_clusters, i) {
			repeat (NUM_DIMS, dim) {
				new_centroids[i][dim] = 0.0;
			}
			cluster_counts[i] = 0u;
		}
		
		// Assign each point to the nearest centroid.
		repeat (num_vectors, i) {
			const int* vector = vectors[i];
			double min_dist = DBL_MAX;
			unsigned int best_centroid_label = 0;
			
			// Find nearest centroid.
			repeat (num_clusters, j) {
				double dist = sparse_dif_c(vector, centroids[j]);
				if (dist < min_dist) {
					min_dist = dist;
					best_centroid_label = j;
				}
			}
			
			// Update label to new centroid, if necessary.
			if (labels[i] != best_centroid_label) {
				labels[i] = best_centroid_label;
				changed = true;
			}
			
			// Accumulate values for new centroid calculation.
			double* best_centroid = new_centroids[best_centroid_label];
			for (unsigned int i = 0u, dim = 0u; dim < NUM_DIMS;) {
				const int val = vector[i++];
				if (val < 0) dim += (unsigned)(-val);
				else best_centroid[dim++] += (double)val;
			}
			cluster_counts[best_centroid_label]++;
		}
		
		// Save number of iterations (if needed).
		if (iterations != NULL) *iterations = iter + 1;
		
		// Stop if centroids didn't change.
		if (!changed) break;
		
		// Update centroids.
		repeat (num_clusters, i) {
			if (cluster_counts[i] == 0u) continue;
			double* centroid = centroids[i];
			const double* new_centroid = new_centroids[i];
			const unsigned int cluster_count = cluster_counts[i];
			repeat (NUM_DIMS, dim) {
				centroid[dim] = new_centroid[dim] / cluster_count;
			}
		}
		
		// Print cluster size for debugging.
		const double average_cluster_size = get_cluster_size(vectors, num_vectors, labels, centroids, num_clusters);
		
		// Is there enough improvement?
		const double improvement = previous_average_cluster_size - average_cluster_size;
		fprintf(kmeans_file, INDENT"Improvement: %.4lf\n", improvement);
		fflush(kmeans_file);
		if (improvement < KMEANS_IMPROVEMENT_THRESHOLD) break;
		previous_average_cluster_size = average_cluster_size;
	}
	
	// kmeans memory usage.
	fprint_mem(stdout);
	
	// Free memory.
	repeat (num_clusters, i) {
		free(new_centroids[i]);
	}
	free(new_centroids);
	free(cluster_counts);
}

/*** Loads the dataset from disk into the global `dataset` array.
 *** 
 *** @attention - Not thread safe: Uses strdup and strtok.
 *** 
 *** @param dataset_path Path to a comma-separated file containing the dataset strings.
 ***/
void load_dataset(const char* dataset_path, char** dataset, const unsigned int dataset_size, bool phone) {
	FILE *file = fopen(dataset_path, "r");
	if (!file) {
		char error_buf[BUFSIZ];
		snprintf(error_buf, sizeof(error_buf), "Could not open \"%s\"", dataset_path);
		fail(error_buf);
	}
	
	char* buffer = NULL;
	size_t bufsize = 0;
	int error = getdelim(&buffer, &bufsize, EOF, file) == -1;
	fclose(file);
	
	if (error) {
		char error_buf[BUFSIZ];
		snprintf(error_buf, sizeof(error_buf), "Could not read \"%s\"", dataset_path);
		fail(error_buf);
	}
	
	// Verify dataset size.
	size_t size = 1;
	for (char* buf = buffer; *buf != '\0';) {
        if (*buf == ',') size++;
        buf++;
    }
	if (size != dataset_size) {
		fprintf(stderr, "\nWarning: Expected dataset of size %u but got %ld.\n\n", dataset_size, size);
	}
	
	char* token = strtok(buffer, ",");
	for (size_t count = 0; token && count < dataset_size; token = strtok(NULL, ",")) {
		if (phone) {
			/** Verify length can be a valid phone number. **/
			const size_t len = strlen(token);
			if (len < 10u) {
				printf("Phone number too short: %s.\n", token);
				continue;
			}
			if (len > 18u) {
				printf("Phone number too long: %s.\n", token);
				continue;
			}
			
			/** Parse phone number. **/
			char buf[11u], cur_char = token[0];
			unsigned int j = ((cur_char == '+') ? 2u :
			                 ((cur_char == '1') ? 1u : 0u));
			unsigned int number_len = 0u;
			for (cur_char = token[j++]; cur_char != '\0' && number_len <= 10u; cur_char = token[j++]) {
				if (
					cur_char == '-' ||
					cur_char == ' ' ||
					cur_char == '(' ||
					cur_char == ')'
				) continue;
				else if (!isdigit(cur_char)) {
					/** Unknown character. **/
					printf("Unknown character '%c' %d in phone number: %s.\n", cur_char, cur_char, token);
					goto next_phone_number;
				}
				
				/** Add the character to the phone number. */
				buf[number_len++] = cur_char;
			}
			
			/** Check number of digits. **/
			if (number_len < 10u) {
				printf("Phone number has %u < 10 digits: %s.\n", number_len, token);
				continue;
			}
			if (number_len > 10u) {
				printf("Phone number has %u > 10 digits: %s.\n", number_len, token);
				continue;
			}
			
			/** Copy valid phone number (with no null-terminator). **/
			char* cur = (char*)mallocs(10u * sizeof(char*));
			memcpy(cur, buf, 10u);
			dataset[count++] = cur;
			
			next_phone_number:;
		} else {
			dataset[count++] = strdup(token);
		}
	}
	
	free(buffer);
}

typedef struct {
	unsigned int i, j;
} Dup;

char** global_dataset;
/*** Scans the entire dataset for duplicates by pairwise similarity.
 *** 
 *** This is the "complete" strategy: every pair `(i, j)` is compared and
 *** pairs with `(similarity > DUPE_THRESHOLD)` are appended to the returned list.
 *** The function also logs found duplicates to the `complete_file`.
 *** 
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @param num_vectors The number of vectors to be scanned.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O(n^2 * 3d)`
 ***/
ArrayList* find_complete_dups(int** vectors, unsigned int num_vectors) {
	ArrayList* complete_dups = al_newc(num_vectors * 2u);
	for (unsigned int i = 0u; i < num_vectors; i++) {
		const int* v1 = vectors[i];
		for (unsigned int j = i + 1; j < num_vectors; j++) {
			const int* v2 = vectors[j];
			const double sim = sparse_similarity(v1, v2);
			if (sim > DUPE_THRESHOLD) {
				Dup* dup = (Dup*)mallocs(sizeof(Dup));
				dup->i = i;
				dup->j = j;
				al_add(complete_dups, dup);
				// fprintf(kmeans_file, "%s,%s,%.8lf\n", global_dataset[i], global_dataset[j], sim);
			}
		}	
	}
	
	return complete_dups;
}

typedef struct {
	int** vectors;
	unsigned int i;
	unsigned int num_vectors;
	unsigned int num_threads;
	ArrayList* dups;
	pthread_mutex_t* work_mutex;
	pthread_mutex_t* list_mutex;
} WorkStruct;

static void* complete_search_worker(void* argp) {
	// Extract globals.
	WorkStruct* work = (WorkStruct*)argp;
	int** vectors = work->vectors;
	const unsigned int num_vectors = work->num_vectors;
	const unsigned int num_threads = work->num_threads;
	ArrayList* dups = work->dups;
	pthread_mutex_t* work_mutex = work->work_mutex;
	pthread_mutex_t* list_mutex = work->list_mutex;
	
	// Work loop.
	while (true) {
		// Get work.
		pthread_mutex_lock(work_mutex);
		const unsigned int remaining_work = num_vectors - work->i;
		if (remaining_work == 0) {
			pthread_mutex_unlock(work_mutex);
			break;
		}
		const unsigned int my_work = max(remaining_work / (num_threads * 4u), min(256u, remaining_work));
		const unsigned int start_i = work->i;
		const unsigned int end_i = start_i + my_work;
		work->i = end_i;
		pthread_mutex_unlock(work_mutex);
		
		// Do work.
		for (unsigned int i = start_i; i < end_i; i++) {
			const int* v1 = vectors[i];
			for (unsigned int j = i + 1; j < num_vectors; j++) {
				const int* v2 = vectors[j];
				const double sim = sparse_similarity(v1, v2);
				if (sim > DUPE_THRESHOLD) {
					Dup* dup = (Dup*)mallocs(sizeof(Dup));
					dup->i = i;
					dup->j = j;
					
					pthread_mutex_lock(list_mutex);
					al_add(dups, dup);
					// if (kmeans_file) {
					// 	fprintf(kmeans_file, "%s,%s,%lf\n",
					// 		global_dataset[i], global_dataset[j], sim);
					// }
					pthread_mutex_unlock(list_mutex);
				}
			}
		}
	}
	
	return NULL;
}

ArrayList* find_complete_dups_par(int** vectors, unsigned int num_vectors) {
	// Detect available cores to determine how many threads to use.
	long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
	if (num_cores < 1) num_cores = 1;
	unsigned int num_threads = (unsigned int)num_cores;
	if (num_threads > num_vectors) num_threads = num_vectors;
	if (num_threads < 1) num_threads = 1;
	printf(INDENT"Detected: %ld cores, using %d threads...\n", num_cores, num_threads);
	
	// Set up thread data structures.
	ArrayList* dups = al_newc((size_t)num_vectors * 2u);
	pthread_t* threads = mallocs(num_threads * sizeof(pthread_t));
	WorkStruct* work = mallocs(sizeof(WorkStruct));
	pthread_mutex_t work_mutex, list_mutex;
	pthread_mutex_init(&work_mutex, NULL);
	pthread_mutex_init(&list_mutex, NULL);
	
	// Initialize work struct.
	work->vectors = vectors;
	work->i = 0u;
	work->num_vectors = num_vectors;
	work->num_threads = num_threads;
	work->dups = dups;
	work->work_mutex = &work_mutex;
	work->list_mutex = &list_mutex;
	
	// Spin up threads.
	for (unsigned int t = 0; t < num_threads; t++) {
		pthread_create(&threads[t], NULL, complete_search_worker, work);
	}
	
	// Join threads.
	for (unsigned int t = 0u; t < num_threads; t++) {
		check(fflush(stdout), "fflush(stdout)");
		pthread_join(threads[t], NULL);
	}
	
	// Cleanup.
	pthread_mutex_destroy(&work_mutex);
	pthread_mutex_destroy(&list_mutex);
	free(threads);
	free(work);
	
	return dups;
}

/*** Finds duplicates using a sliding-window strategy.
 *** 
 *** Compares each vector only with the next n pairs in a "sliding window"
 *** that moves down the data. Results are logged to `sliding_file` and
 *** validated against the `complete_dups` list.
 ***
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @param num_vectors The number of vectors to be scanned.
 *** @param window_size The size of the sliding window.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O(nw * 3d)`
 ***/
ArrayList* find_sliding_dups(int** vectors, unsigned int num_vectors, const unsigned int window_size) {
	ArrayList* sliding_dups = al_newc(512);
	repeat (num_vectors, i) {
		const int* v1 = vectors[i];
		const unsigned int j_max = min(i + window_size, num_vectors);
		for (unsigned int j = i + 1; j < j_max; j++) {
			const int* v2 = vectors[j];
			if (sparse_similarity(v1, v2) > DUPE_THRESHOLD) {
				Dup* dup = (Dup*)mallocs(sizeof(Dup));
				dup->i = i;
				dup->j = j;
				al_add(sliding_dups, dup);
			}
		}
	}
	
	// Lock results.
	al_lock(sliding_dups);
	al_trim_to_size(sliding_dups);
	
	return sliding_dups;
}

/*** Runs k-means clustering then finds duplicates inside clusters.
 ***
 *** Steps:
 *** 
 ***  - Allocate labels and centroids.
 *** 
 ***  - Run kmeans(...) to assign labels.
 *** 
 ***  - For each cluster, compare points inside the cluster and record pairs
 ***    with `(similarity > DUPE_THRESHOLD)`.
 *** 
 ***  - Log duplicates and cluster contents to `kmeans_file`, validate found
 ***    pairs against `complete_dups` to detect mistakes.
 ***
 *** @param vectors Array of precomputed frequency vectors for all dataset strings.
 *** @param num_vectors The number of vectors to be scanned.
 *** @param iterations The number of iterations that actually executed is stored
 *** 	here. Leave this NULL if you don't care.
 *** @param max_iter Maximum iterations passed to kmeans().
 *** @param num_clusters Number of clusters to produce.
 *** @returns A locked ArrayList containing duplicate index pairs.
 *** 
 *** Complexity: `O((nk + nd) + 3d*s^2)`
 *** Complexity: `O((nk + nd) + 3d*(n/k)^2)`
 *** Complexity: `O(nk + (n/k)^2)` (ignoring i & d)
 ***/
ArrayList* find_kmeans_dups(int** vectors, unsigned int num_vectors, unsigned int* iterations, unsigned int max_iter, unsigned int num_clusters) {
	// Create timers.
	Timer* timer_total = timer_new();
	Timer* timer_clustering = timer_new();
	Timer* timer_scan = timer_new();
	timer_start(timer_total);
	
	// Malloc memory for finding clusters.
	unsigned int* labels = (unsigned int*)mallocs(num_vectors * sizeof(unsigned int));
	repeat (num_vectors, i) labels[i] = 0;
	double** centroids = (double**)mallocs(num_clusters * sizeof(double*));
	repeat (num_clusters, i) centroids[i] = (double*)mallocs(NUM_DIMS * sizeof(double));
	fprintf(kmeans_file, "\nMalloced...\n");
	fflush(kmeans_file);
	
	// Execute kmeans clustering.
	double debug_time = 0.0;
	timer_benchmark(timer_clustering,
		kmeans(vectors, num_vectors, labels, centroids, iterations, max_iter, num_clusters);
	);
	fprintf(kmeans_file, "Clustered...\n");
	fflush(kmeans_file);
	
	// Find duplocates in clusters.
	timer_benchmark(timer_scan,
		ArrayList* kmeans_dups = al_newc(512);
		repeat (num_vectors, i) {
			const int* v1 = vectors[i];
			const unsigned int label = labels[i];
			for (unsigned int j = i + 1; j < num_vectors; j++) {
				if (labels[j] != label) continue;
				const int* v2 = vectors[j];
				if (sparse_similarity(v1, v2) > DUPE_THRESHOLD) {
					Dup* dup = (Dup*)mallocs(sizeof(Dup));
					dup->i = i;
					dup->j = j;
					al_add(kmeans_dups, dup);
				}
			}
		}
	);
	
	fprintf(kmeans_file, "Checked!\n");
	fflush(kmeans_file);
	
	// Lock results.
	al_lock(kmeans_dups);
	al_trim_to_size(kmeans_dups);
	
	// Print benchmarks.
	timer_stop(timer_total);
	const double total_time = timer_get(timer_total);
	timer_print_cmp(timer_clustering, INDENT"Kmeans clustering", total_time);
	timer_print_cmp(timer_scan, INDENT"Kmeans checking", total_time);
	printf(INDENT"Kmeans debug time: %.4lfs.\n", debug_time);
	
	// Free memory.
	free(labels);
	repeat (num_clusters, i) {
		free(centroids[i]);
	}
	free(centroids);
	timer_free(timer_total);
	timer_free(timer_clustering);
	timer_free(timer_scan);
	
	return kmeans_dups;
}

ArrayList* test_complete_search(int** vectors, unsigned int num_vectors, bool par, Timer* timer) {
	// Flush buffers to reduce flush overhead during benchmark.
	printf("\nComplete search %son %u records:\n", (par) ? "(in parallel) " : "", num_vectors);
	check(fflush(stdout), "fflush(stdout)");
	
	// Execute the complete search dupe detection.
	timer_benchmark(timer,
		ArrayList* complete_dups = (par)
			? find_complete_dups_par(vectors, num_vectors)
			: find_complete_dups(vectors, num_vectors);
	);
	check(fflush(stdout), "fflush(stdout)");
	
	// Lock results.
	al_lock(complete_dups);
	al_trim_to_size(complete_dups);
	
	// Print complete search summary.
	printf(INDENT"Dups: %ld\n", complete_dups->size);
	printf(INDENT"Time: %.4lfs\n", timer_store(timer));
	check(fflush(stdout), "fflush(stdout)");
	
	return complete_dups;
}

void test_sliding_search(int** vectors, unsigned int num_vectors, Timer* timer, ArrayList* complete_dups) {
	// Sliding window with various window sizes.
	size_t num_window_sizes = sizeof(window_sizes) / sizeof(window_sizes[0]);
	repeat (num_window_sizes, i) {
		unsigned int window_size = window_sizes[i];
		
		// Flush buffers to reduce flush overhead during benchmark.
		printf("\nSliding window (x%u):\n", window_size);
		check(fflush(stdout), "fflush(stdout)");
			
		// Execute sliding window dupe detection.
		timer_benchmark(timer,
			ArrayList* sliding_dups = find_sliding_dups(vectors, num_vectors, window_size);
		);
		double sliding_time = timer_get(timer);
		
		// Print sliding summary.
		double percent_success_sliding = 100.0 * (double)sliding_dups->size / (double)complete_dups->size;
		printf(INDENT"Dups: %ld/%ld (%%%.2lf)\n", sliding_dups->size, complete_dups->size, percent_success_sliding);
		const double time_percent = 100.0f * sliding_time / timer->stored_duration;
		printf(INDENT"Time: %.4lfs (%%%.2f)\n", sliding_time, time_percent);
		check(fflush(stdout), "fflush(stdout)");
		
		// Free memory.
		al_free(sliding_dups);
	}
}

void test_kmeans_search(int** vectors, unsigned int num_vectors, unsigned int* cluster_counts, unsigned int num_cluster_counts, Timer* timer, ArrayList* complete_dups) {
	// Kmeans with various k values.
	repeat (num_cluster_counts, i) {
		unsigned int cluster_count = cluster_counts[i], iterations = 0;
		
		// Flush buffers to reduce flush overhead during benchmark.
		fprintf(kmeans_file, "\n\n============================\n%u vectors, %u clusters\n", num_vectors, cluster_count);
		printf("\nKmeans (num_clusters=%u):\n", cluster_count);
		check(fflush(stdout), "fflush(stdout)");
		
		// Execute kmeans dupe detection.
		timer_benchmark(timer,
			ArrayList* kmeans_dups = find_kmeans_dups(vectors, num_vectors, &iterations, max_iter, cluster_count);
		);
		double kmeans_time = timer_get(timer);
	
		// Print sliding summary.
		const double time_percent = 100.0f * kmeans_time / timer->stored_duration;
		const double percent_success_kmeans = 100.0 * (double)kmeans_dups->size / (double)complete_dups->size;
		printf(INDENT"Time: %.4lfs (%%%.2f)\n", kmeans_time, time_percent);
		printf(INDENT"Iterations: %u/%u\n", iterations, max_iter);
		printf(INDENT"Dups: %ld/%ld (%%%.2lf)\n", kmeans_dups->size, complete_dups->size, percent_success_kmeans);
		check(fflush(stdout), "fflush(stdout)");
		
		// Free memory.
		al_free(kmeans_dups);
	}
}

// 100,000 records should take almost 10 minutes on LightSys hardware.
// Any more than 10 minutes, and we go to kmeans for faster execution time.
#define MAX_COMPLETE_SEARCH 0 // 100 * 1000
void test_lightning_search(int** vectors, unsigned int num_vectors, Timer* timer, ArrayList* complete_dups) {
	printf("\nLightning Search:\n");
	printf(INDENT"Data Size: %u\n", num_vectors);
	
	ArrayList* dups;
	if (num_vectors <= MAX_COMPLETE_SEARCH) {
		check(fflush(stdout), "fflush(stdout)");
		timer_benchmark(timer,
			dups = find_complete_dups(vectors, num_vectors);
		);
	} else {
		unsigned int cluster_count = compute_k(num_vectors), iterations = 0;
		printf(INDENT"Clusters: %u\n", cluster_count);
		check(fflush(stdout), "fflush(stdout)");
		
		// Execute kmeans dupe detection.
		timer_benchmark(timer,
			dups = find_kmeans_dups(vectors, num_vectors, &iterations, max_iter, cluster_count);
		);
		printf(INDENT"Iterations: %u/%u\n", iterations, max_iter);
	}
	
	// Early cleanup
	const size_t num_dups = dups->size;
	al_free(dups);

	// Print summary.
	const double time = timer_get(timer);
	const double time_percent = 100.0f * time / timer->stored_duration;
	const double percent_success = 100.0 * (double)num_dups / (double)complete_dups->size;
	printf(INDENT"Time: %.4lfs (%%%.2f)\n", time, time_percent);
	printf(INDENT"Dups: %ld/%ld (%%%.2lf)\n", num_dups, complete_dups->size, percent_success);
	check(fflush(stdout), "fflush(stdout)");
}

#define PHONE_LEN 10u
unsigned int exp_fn_edit_dist(const char* str1, const char* str2)
    {
    /*** lev_matrix:
     *** For all i and j, d[i][j] will hold the Levenshtein distance between
     *** the first i characters of s and the first j characters of t.
     *** 
     *** As they say, no dynamic programming algorithm is complete without a
     *** matrix that you fill out and it has the answer in the final location.
     ***/
    unsigned int lev_matrix[PHONE_LEN + 1][PHONE_LEN + 1];
    
    /*** Base case #0:
     *** Transforming an empty string into an empty string has 0 cost.
     ***/
    lev_matrix[0][0] = 0u;
    
    /*** Base case #1:
     *** Any source prefixe can be transformed into an empty string by
     *** dropping each character.
     ***/
    for (unsigned int i = 1u; i <= PHONE_LEN; i++)
		lev_matrix[i][0] = i;
    
    /*** Base case #2:
     *** Any target prefixes can be transformed into an empty string by
     *** inserting each character.
     ***/
    for (unsigned int j = 1u; j <= PHONE_LEN; j++)
		lev_matrix[0][j] = j;
    
    /** General Case **/
    for (unsigned int i = 1u; i <= PHONE_LEN; i++)
	{
	for (unsigned int j = 1u; j <= PHONE_LEN; j++)
	    {
	    /** Equal characters need no changes. **/
	    if (str1[i - 1] == str2[j - 1])
		lev_matrix[i][j] = lev_matrix[i - 1][j - 1];
	    
	    /*** We need to make a change, so use the opereration with the
	     *** lowest cost out of delete, insert, replace, or swap.
	     ***/
	    else 
		{
		unsigned int cost_delete  = lev_matrix[i - 1][j] + 1u;
		unsigned int cost_insert  = lev_matrix[i][j - 1] + 1u;
		unsigned int cost_replace = lev_matrix[i-1][j-1] + 1u;
		
		/** If a swap is possible, calculate the cost. **/
		bool can_swap = (
		    i > 1 && j > 1 &&
		    str1[i - 1] == str2[j - 2] &&
		    str1[i - 2] == str2[j - 1]
		);
		unsigned int cost_swap = (can_swap) ? lev_matrix[i - 2][j - 2] + 1 : UINT_MAX;
		
		// Find the best operation.
		lev_matrix[i][j] = min(min(min(cost_delete, cost_insert), cost_replace), cost_swap);
		}
	    }
	}
    
    return lev_matrix[PHONE_LEN][PHONE_LEN];
    }

ArrayList* find_phone_dups(char** strs, unsigned int num_vectors) {
	ArrayList* phone_dups = al_newc(num_vectors * 2u);
	repeat (num_vectors, i) {
		const char* str1 = strs[i];
		for (unsigned int j = i + 1; j < num_vectors; j++) {
			const char* str2 = strs[j];
			const double sim = (double)(10u - exp_fn_edit_dist(str1, str2)) / 10.0;
			if (sim > DUPE_THRESHOLD) {
				Dup* dup = (Dup*)mallocs(sizeof(Dup));
				dup->i = i;
				dup->j = j;
				al_add(phone_dups, dup);
				fprintf(kmeans_file, "%s,%s,%lf\n", str1, str2, sim);
			}
		}	
	}
	
	return phone_dups;
}

ArrayList* test_phone_search(char** strs, unsigned int num_strs, Timer* timer) {
	// Flush buffers to reduce flush overhead during benchmark.
	printf("\nPhone search on %u records:\n", num_strs);
	check(fflush(stdout), "fflush(stdout)");
	
	// Execute the complete search dupe detection.
	fflush(kmeans_file);
	timer_benchmark(timer,
		ArrayList* phone_dups = find_phone_dups(strs, num_strs);
	);
	fflush(kmeans_file);
	
	// Lock results.
	al_lock(phone_dups);
	al_trim_to_size(phone_dups);
	
	// Print complete search summary.
	printf(INDENT"Dups: %ld\n", phone_dups->size);
	printf(INDENT"Time: %.4lfs\n", timer_store(timer));
	check(fflush(stdout), "fflush(stdout)");
	
	return phone_dups;
}

double cmp_sim(const char* str1, const char* str2) {
	size_t len = 2, sz = 32;
	char strs[len][sz];
	snprintf(memset(strs[0], 0, sz), sz, "%s", str1);
	snprintf(memset(strs[1], 0, sz), sz, "%s", str2);
	
	int** vectors = (int**)mallocs(len * sizeof(int*));
	repeat (len, i) vectors[i] = build_vector(strs[i]);
	double dif = sparse_similarity(vectors[0], vectors[1]);
	// printf("%s =? %s: %lf\n", strs[0], strs[1], dif);
	free(vectors);
	return dif;
}

// void test_sims(void) {
// 	unsigned int num_records = 4;
// 	static char* data[][6] = {
//         {"John", "JHAN", "Smith", "SM0XMT", "JohnSmith", "1284 Lecross Street 58214"},
//         {"Jain", "JHAN", "Schmid", "XMTSMT", "JainDoe", "3614 Feron Place 62841"},
//         {"Bart", "PRT", "Smith", "SM0XMT", "ImOnTopOfTheWorldYeah", "1284 Lecross Street 58214"},
//         {"John", "JHAN", "Smith", "SM0XMT", "JohnSmtih", "1283 Lekroce Streat 58124"},
//     };
// 	char*** dataset = (char***)data;
	
// 	repeat (num_records, i) repeat (i, j) {
// 		double fname  = cmp_sim(data[i][0], data[j][0]);
// 		double fnamem = cmp_sim(data[i][1], data[j][1]);
// 		double lname  = cmp_sim(data[i][2], data[j][2]);
// 		double lnamem = cmp_sim(data[i][3], data[j][3]);
// 		double email  = cmp_sim(data[i][4], data[j][4]);
// 		double addr   = cmp_sim(data[i][5], data[j][5]);
		
// 		double fname_ag = max(fname, fnamem * 0.9);
// 		double lname_ag = max(lname, lnamem * 0.9);
		
// 		double ave = (fname_ag + lname_ag + email + addr) / 4.0;
// 		double smt = (fname_ag * lname_ag) * 0.6 + email * 0.2 + addr * 0.2;
		
// 		printf(
// 			"%s %s =? %s %s -> %lf %lf\n",
// 			data[j][0], data[j][2], data[i][0], data[i][2],
// 			ave, smt
// 		);
// 		printf(
// 			INDENT"%lf (%lf), %lf (%lf), %lf, %lf\n",
// 			fname, fnamem, lname, lnamem, email, addr
// 		);
// 	}
// }

int dup_cmp(const void *pa, const void *pb) {
	// pa and pb are pointers to elements in the array; each element is a Dup*.
    const Dup* a = *(const Dup* const*)pa;
    const Dup* b = *(const Dup* const*)pb;

	if (a->i < b->i) return -1;
	if (a->i > b->i) return  1;
	if (a->j < b->j) return -1;
	if (a->j > b->j) return  1;
	return 0;
}

/*** Program entry point: runs various tests and reports results.
 *** 
 *** Usage:
 *** 
 ***  - argv[1] -> kmeans_file_name
 ***
 *** The program:
 *** 
 ***  - opens the three output files,
 *** 
 ***  - loads the dataset and builds vectors,
 *** 
 ***  - runs: complete search, sliding window, and kmeans clustering,
 *** 
 ***  - prints timing and success statistics, then frees resources and exits.
 ***
 *** @param argc Number of command-line arguments; must be 4.
 *** @param argv Command-line argument vector.
 *** @returns `0` on success, `1` on incorrect invocation.
 ***/
int main(int argc, char* argv[]) {
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <kmeans_file_name>\n", argv[0]);
		return 1;
	}
	kmeans_file = fopen(argv[1], "w");
	Timer* timer = timer_new();
	Timer* timer_total = timer_new();
	timer_start(timer_total);
	
	// printf("similarity: %lf", cmp_sim("error","undefined"));
	// goto main_end;
	
	// Set buffers to only flush manually for more accurate performance evaluation.
	// setvbuf(stdout, NULL, _IOFBF, (2 * 1000 * 1000));
	setvbuf(kmeans_file, NULL, _IOFBF, (4 * 1000 * 1000));
	printf("Begin!\n");
	
	// Print basic settings info.
	printf(
		"\nSettings:\n"
			INDENT"Dimensions: %u\n"
			INDENT"Dupe Threshold: %.4f\n",
		NUM_DIMS,
		DUPE_THRESHOLD
	);
	fprint_mem(stdout);
	check(fflush(stdout), "fflush(stdout)");
	
	size_t num_dataset_sizes = sizeof(dataset_sizes) / sizeof(dataset_sizes[0]);
	repeat (num_dataset_sizes, size_index) {
		unsigned int dataset_size = dataset_sizes[size_index];
		
		// Load dataset and build vectors.
		char** dataset = global_dataset = (char**)mallocs(dataset_size * sizeof(char*));
		char path[BUFSIZ];
		snprintf(path, sizeof(path), PATH_FORMAT, dataset_size);
		
		timer_start(timer);
		load_dataset(path, dataset, dataset_size, false);
		int** vectors = (int**)mallocs(dataset_size * sizeof(int*));
		repeat (dataset_size, i) {
			vectors[i] = build_vector(dataset[i]);
		}
		timer_stop(timer);
		
		// Print
		printf("\n\nDataset Loaded (x%u):\n", dataset_size);
		timer_print(timer, INDENT"Loading");
		fprint_mem(stdout);
		check(fflush(stdout), "fflush(stdout)");
		
		// Complete search.
		ArrayList* complete_dups_par = (dataset_size <= 100000)
			? test_complete_search(vectors, dataset_size, true, timer)
			: al_new();
			
		ArrayList* complete_dups = (dataset_size <= 100000)
			? test_complete_search(vectors, dataset_size, false, timer)
			: al_new();
		
		printf("\n");
		bool success = true;
		if (complete_dups_par->size != complete_dups->size) {
			success = false;
			printf("Size mismatch: %lu, but should be %lu.\n", complete_dups_par->size, complete_dups->size);
			check(fflush(stdout), "fflush(stdout)");
		}
		
		printf("Sorting...\n");
		check(fflush(stdout), "fflush(stdout)");
		al_sort(complete_dups_par, dup_cmp);
		al_sort(complete_dups, dup_cmp);
		
		for (size_t i = 0u; i < complete_dups_par->size; i++) {
			const unsigned int pari = ((Dup*)complete_dups_par->data[i])->i;
			const unsigned int parj = ((Dup*)complete_dups_par->data[i])->j;
			const unsigned int cori = ((Dup*)complete_dups->data[i])->i;
			const unsigned int corj = ((Dup*)complete_dups->data[i])->j;
			
			if (pari != cori || parj != corj) {
				success = false;
			    printf("Value mismatch: (%u, %u) #%lu should be (%u, %u).\n", pari, parj, i, cori, corj);
			    check(fflush(stdout), "fflush(stdout)");
			}
		}
		
		if (success) {
			printf("Success!\n");
			check(fflush(stdout), "fflush(stdout)");
		} else {
			printf("Failure!\n");
			check(fflush(stdout), "fflush(stdout)");
		}

		// Sliding search.
		// test_sliding_search(vectors, dataset_size, timer, complete_dups);
		
		// Lightning search.
		// test_lightning_search(vectors, dataset_size, timer, complete_dups);
		
		// Phone search.
		// test_phone_search(dataset, dataset_size, timer);
		
		// Clean up.
		repeat (dataset_size, i) {
			free(vectors[i]);
			free(dataset[i]);
		}
		free(vectors);
		free(dataset);
		repeat (complete_dups_par->size, i) {
			free(complete_dups_par->data[i]);
			free(complete_dups->data[i]);
		}
		al_free(complete_dups_par);
		al_free(complete_dups);
		check(fflush(stdout), "fflush(stdout)");
	}
	
	// Print the total execution time.
	timer_stop(timer_total);
	timer_print(timer_total, "\nTotal");
	
	// End program and flush all buffers.
	printf("\nDone!\n");
	check(fflush(stdout), "fflush(stdout)");
	
	// Close files.
	if (kmeans_file) fclose(kmeans_file);
	
	// Free memory.
	timer_free(timer);
	timer_free(timer_total);
	
	return 0;
}
