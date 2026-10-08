#include <stdio.h>
#include <stdlib.h>

#include "../include/read_file_lib.h"
#include "../include/spmv_type.h"

void read_from_file_and_init(char *file_path, struct COO *coo_data) {
  FILE *file;
  const size_t BUFFER_SIZE = 1 * 1024 * 1024;

  file = fopen(file_path, "r");
  if (!file) {
    perror("Error opening file!\n");
    exit(1);
  }

  char *buffer = (char *)malloc(BUFFER_SIZE);
  if (!buffer) {
    perror("Failed to allocate file buffer");
    fclose(file);
    exit(1);
  }
  setvbuf(file, buffer, _IOFBF, BUFFER_SIZE);

  char line_buffer[1024];
  while (fgets(line_buffer, sizeof(line_buffer), file)) {
    if (line_buffer[0] != '%')
      break;
  }

  int n, m, nnz;
  if (sscanf(line_buffer, "%d %d %d", &n, &m, &nnz) != 3) {
    perror("Error reading graph metadata");
    free(buffer);
    fclose(file);
    exit(-1);
  }
  coo_data->num_rows = n;
  coo_data->num_cols = m;
  coo_data->num_non_zeros = nnz;

  coo_data->a_val = (dtype *)malloc(nnz * sizeof(dtype));
  coo_data->a_row = (int *)malloc(nnz * sizeof(int));
  coo_data->a_col = (int *)malloc(nnz * sizeof(int));

  if (!coo_data->a_val || !coo_data->a_row || !coo_data->a_col) {
    perror("Failed to allocate memory for COO matrix data");
    free(buffer);
    fclose(file);
    free(coo_data->a_val);
    free(coo_data->a_row);
    free(coo_data->a_col);
    exit(1);
  }

  int r, c;
  dtype v;
  for (int i = 0; i < nnz; i++) {
    if (fscanf(file, "%d %d %f", &r, &c, &v) != 3) {
      fprintf(stderr, "Error reading entry %d\n", i);
      free(buffer);
      fclose(file);
      free(coo_data->a_val);
      free(coo_data->a_row);
      free(coo_data->a_col);
      exit(1);
    }
    coo_data->a_row[i] = r - 1;
    coo_data->a_col[i] = c - 1;
    coo_data->a_val[i] = v;
  }

  free(buffer);
  fclose(file);
}
