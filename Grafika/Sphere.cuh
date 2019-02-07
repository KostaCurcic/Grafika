#pragma once

#include "Vector.cuh"

class Sphere
{
public:
	__host__ __device__ Sphere();
	__host__ __device__ Sphere(Point C, float R);
	__host__ __device__ Vector Normal(Point&) const;

	Point c;
	float r;

private:

};