#include "Sphere.cuh"

__host__ __device__ Sphere::Sphere()
{
	c = Point(0, 0, 0);
	r = 0;
}

__host__ __device__ Sphere::Sphere(Point C, float R)
{
	c = C;
	r = R;
}

__host__ __device__ Vector Sphere::Normal(Point &p) const
{
	return ((Vector)(p - c)).Normalize();
}
