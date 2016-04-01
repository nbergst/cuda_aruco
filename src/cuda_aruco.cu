#include "cuda_aruco.h"

#include "cudautils.h"



#include <vector>

std::vector<cudaStream_t> streams;

struct cbs {
    unsigned char* src;
    unsigned char* dst;
    int width;
    int height;
    int pitch;
};

void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){

    cbs* s = (cbs*)data;

    for (int y=0; y<s->height; ++y) {
        memcpy(s->dst + y*s->width, s->src + y*s->pitch, s->width);
    }

    free(data);
}


void cuda_aruco::CudaProcessor::createBuffer(int _width, int _height,
                                             std::vector<CudaImage>& _buffers,
                                             int _nimages) {
  width = _width;
  height = _height;
  cudaMallocPitch(&buffer, &pitch, width, height * _nimages);

  _buffers.resize(_nimages);

  for (int i = 0; i < _nimages; ++i) {
      _buffers[i].width = width;
      _buffers[i].height = height;
      _buffers[i].pitch = pitch;
      cudaMallocHost(&(_buffers[i].h_data), height*width);
      _buffers[i].d_data = &buffer[i * height * pitch];
      _buffers[i].pitch_h = width;
  }

  streams.resize(5);
  for (int i=0; i<5; ++i) {
      cudaStreamCreate(&streams[i]);
  }

}

void cuda_aruco::CudaProcessor::mallocPitch(void** d, size_t* p, size_t w, size_t h) {
    cudaMallocPitch(d,p,w,h);
  }


void cuda_aruco::CudaProcessor::readbackAndCopy(CudaImage& _src, unsigned char* _dst, int _stream) {

    cudaMemcpy2DAsync(_src.h_data, _src.pitch, _src.d_data, _src.pitch, _src.width, _src.height, cudaMemcpyDeviceToHost, streams[_stream]);

    cbs* s = (cbs*)malloc(sizeof(cbs));
    s->src = _src.h_data;
    s->dst = _dst;
    s->height = _src.height;
    s->width = _src.width;
    s->pitch = _src.pitch;

    cudaStreamAddCallback(streams[_stream], MyCallback, (void*)s, 0);
}


void cuda_aruco::CudaProcessor::freeBuffer() { cudaFree(buffer); }



void cuda_aruco::CudaProcessor::allocHost(void** _buff, size_t _bytes) {
  cudaMallocHost(_buff, _bytes);
}



__global__ void filter_kernel_row(const unsigned char* _src,
                                  float* _dst, int _width, int _height,
                                  int _pitch, size_t _pp) {

  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  float p0 = 0.f;
  float p1 = 0.f;
  float p2 = 0.f;
  float p3 = 0.f;
  float p4 = 0.f;

  for (int dy = -6; dy <= 6; ++dy) {
    int ty = y + dy;
    ty = (ty < 0 ? -ty : (ty >= _height ? _height - 1 : ty));
    float val = _src[ty * _pitch + x];
    p4 += val;
    if (abs(dy) < 6) p3 += val;
    if (abs(dy) < 5) p2 += val;
    if (abs(dy) < 4) p1 += val;
    if (abs(dy) < 3) p0 += val;
  }

  _dst[0 * _pp * _height + y * _pp + x] = p0;
  _dst[1 * _pp * _height + y * _pp + x] = p1;
  _dst[2 * _pp * _height + y * _pp + x] = p2;
  _dst[3 * _pp * _height + y * _pp + x] = p3;
  _dst[4 * _pp * _height + y * _pp + x] = p4;
    
}



__global__ void filter_kernel_col(const float* _src,
                                  unsigned char* _dst, int _width, int _height,
                                  int _pitch, size_t _pp) {
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  float p0 = 0.f;
  float p1 = 0.f;
  float p2 = 0.f;
  float p3 = 0.f;
  float p4 = 0.f;

  for (int dx = -6; dx <= 6; ++dx) {
    int tx = x + dx;
    tx = (tx < 0 ? -tx : (tx >= _width ? _width - 1 : tx));
    p4 += _src[4 * _pp * _height + y * _pp + tx];
    if (abs(dx) < 6) p3 += _src[3 * _pp * _height + y * _pp + tx];
    if (abs(dx) < 5) p2 += _src[2 * _pp * _height + y * _pp + tx];
    if (abs(dx) < 4) p1 += _src[1 * _pp * _height + y * _pp + tx];
    if (abs(dx) < 3) p0 += _src[0 * _pp * _height + y * _pp + tx];
  }

  _dst[0*_pitch*_height + y*_pitch+x] = (unsigned char)(p0 * 1.f / (5.f * 5.f) + .5f);
  _dst[1*_pitch*_height + y*_pitch+x] = (unsigned char)(p1 * 1.f / (7.f * 7.f) + .5f);
  _dst[2*_pitch*_height + y*_pitch+x] = (unsigned char)(p2 * 1.f / (9.f * 9.f) + .5f);
  _dst[3*_pitch*_height + y*_pitch+x] = (unsigned char)(p3 * 1.f / (11.f * 11.f) + .5f);
  _dst[4*_pitch*_height + y*_pitch+x] = (unsigned char)(p4 * 1.f / (13.f * 13.f) + .5f);

}



__global__ void filter_kernel(const unsigned char* _src, unsigned char* _dst,
                              int _width, int _height, int _pitch) {
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  float p0 = 0.f;
  float p1 = 0.f;
  float p2 = 0.f;
  float p3 = 0.f;
  float p4 = 0.f;

  if (y < _height && x < _width) {

  for (int dy = -6; dy <= 6; ++dy) {
    int ty = y + dy;
    ty = (ty < 0 ? -ty : (ty >= _height ? _height - 1 : ty));
    for (int dx = -6; dx <= 6; ++dx) {
      int tx = x + dx;
      tx = (tx < 0 ? -tx : (tx >= _width ? _width - 1 : tx));
      p4 += _src[ty * _pitch + tx];
      if (abs(dx) < 6 && abs(dy) < 6) p3 += _src[ty * _pitch + tx];
      if (abs(dx) < 5 && abs(dy) < 5) p2 += _src[ty * _pitch + tx];
      if (abs(dx) < 4 && abs(dy) < 4) p1 += _src[ty * _pitch + tx];
      if (abs(dx) < 3 && abs(dy) < 3) p0 += _src[ty * _pitch + tx];
    }
  }

  p0 *= 1.f / (5.f * 5.f);
  p1 *= 1.f / (7.f * 7.f);
  p2 *= 1.f / (9.f * 9.f);
  p3 *= 1.f / (11.f * 11.f);
  p4 *= 1.f / (13.f * 13.f);

  _dst[0 * _pitch * _height + y * _pitch + x] = (unsigned char)(p0 + .5f);
  _dst[1 * _pitch * _height + y * _pitch + x] = (unsigned char)(p1 + .5f);
  _dst[2 * _pitch * _height + y * _pitch + x] = (unsigned char)(p2 + .5f);
  _dst[3 * _pitch * _height + y * _pitch + x] = (unsigned char)(p3 + .5f);
  _dst[4 * _pitch * _height + y * _pitch + x] = (unsigned char)(p4 + .5f);
  }
}

__global__ void threshold_kernel(const unsigned char* _src, unsigned char* _dst,
                                 int _width, int _height, int _pitch,
                                 double _param) {
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (y < _height && x < _width) {
#pragma unroll
    for (int i = 0; i < 5; ++i)
      _dst[i * _pitch * _height + y * _pitch + x] = 
	_src[y * _pitch + x] > _dst[i * _pitch * _height + y * _pitch + x] - _param
				  ? 0
				  : 255;
  }
}

void cuda_aruco::CudaProcessor::sync(int i) {
    if (i==-1)
        cudaDeviceSynchronize();
    else
        cudaStreamSynchronize(streams[i]);
}

void cuda_aruco::CudaProcessor::cuda_threshold(CudaImage& _src, float* _tmp, size_t _pp,
                                               CudaImage& _dst, double* t1,
                                               double t2) {
  // Set device to prioritize cache
  // cudaFuncSetCacheConfig(filter_kernel, cudaFuncCachePreferL1);

  dim3 blocks(iDivUp(width, 16), iDivUp(height, 16));
  dim3 threads(16, 16);
  filter_kernel_row<<<blocks, threads>>>
    (_src.d_data, _tmp, width, height, pitch, _pp);
  filter_kernel_col<<<blocks, threads>>>
    (_tmp, _dst.d_data, width, height, pitch, _pp);
  /*      filter_kernel<< <blocks, threads>>>
	  (_src.d_data, _dst.d_data, width, height, pitch);*/

  threshold_kernel<<<blocks, threads>>>
    (_src.d_data, _dst.d_data, width, height, pitch, t2);
}
