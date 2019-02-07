#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Point
{
public:
	__host__ __device__ Point();
	__host__ __device__ Point(float, float, float);

	__host__ __device__ Point operator+(const Point&) const;
	__host__ __device__ Point operator-(const Point&) const;

	float x, y, z;

private:
};
