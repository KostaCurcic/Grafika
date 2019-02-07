#include "Point.cuh"

__host__ __device__ Point::Point()
{
	x = y = z = 0;
}

__host__ __device__ Point::Point(float X, float Y, float Z)
{
	x = X;
	y = Y;
	z = Z;
}

__host__ __device__ Point Point::operator+(const Point &p) const
{
	return Point(x + p.x, y + p.y, z + p.z);
}

__host__ __device__ Point Point::operator-(const Point &p) const
{
	return Point(x - p.x, y - p.y, z - p.z);
}