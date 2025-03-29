#include "comm_executor.h"
#include <cassert>
#include <cstdio>

namespace tb {

template <typename T, typename TileLayout>
void allgather_host(T* src_tensor, T* dst_tensor, 
                        int* signals, size_t tensor_size, 
                        int mype, int npes, int div_dim = 0, bool use_pull = false) {
  
  CommExecutor<T, TileLayout, false> comm_executor;
  if (use_pull) {
    printf("Pull mode not supported yet\n");
    assert(false);
  } else {
    for (int dst_pe = 0; dst_pe < npes; dst_pe++) {
        if (dst_pe == mype) continue;
        
        comm_executor.send(dst_tensor, src_tensor, tensor_size, dst_pe, signals);
    }
  }
}

}