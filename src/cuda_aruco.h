#ifndef _CUDA_ARUCO_H_
#define _CUDA_ARUCO_H_

#include <vector>

#include "cudaImage.h"

namespace cuda_aruco {

class CudaProcessor {
 private:
  unsigned char* buffer;
  size_t width;
  size_t height;
  size_t pitch;

 public:
  void createBuffer(int _width, int _height, std::vector<CudaImage>& _buffers, int _nimages);
  void freeBuffer();
  void allocHost(void** _buff, size_t _bytes);
  void sync(int i=-1);
  void cuda_threshold(CudaImage& _src, float* _tmp, size_t _pp,
                      CudaImage &_dst, double *t1,
                      double t2);
  void readbackAndCopy(CudaImage& _src, unsigned char* _dst, int _stream);

  void mallocPitch(void** d, size_t* p, size_t w, size_t h);
   
};

}  // namespace cuda_aruco

#endif  // _CUDA_ARUCO_H_
