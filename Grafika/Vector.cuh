#pragma once

#include "Point.cuh"

class Vector : public Point
{
public:
	__host__ __device__ Vector(const Point&);
	__host__ __device__ Vector() {};
	__host__ __device__ Vector(float X, float Y, float Z) : Point(X, Y, Z) {};
	__host__ __device__ Vector& Unzero();
	__host__ __device__ Vector& Normalize();
	__host__ __device__ float Length() const;
	__host__ __device__ float operator*(const Vector&) const;
	__host__ __device__ Vector operator*(const float) const;

private:

};
