#pragma once

#define CUDA

#ifdef CUDA
	#define DEVICE_PREFIX __host__ __device__

	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
#else
	#define DEVICE_PREFIX 
#endif // CUDA


class Point
{
public:
	DEVICE_PREFIX Point();
	DEVICE_PREFIX Point(float, float, float);

	DEVICE_PREFIX Point operator+(const Point&) const;
	DEVICE_PREFIX Point operator-(const Point&) const;

	float x, y, z;

private:
};
