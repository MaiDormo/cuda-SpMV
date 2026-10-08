#ifndef READ_FILE_LIB_H
#define READ_FILE_LIB_H

#include "spmv_type.h"

#ifdef __cplusplus
extern "C" {
#endif

void read_from_file_and_init(char *file_path, struct COO *coo_data);

#ifdef __cplusplus
}
#endif

#endif
