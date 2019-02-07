#pragma once

#include "Vector.cuh"
#include "Sphere.cuh"


class Ray
{
public:
	__host__ __device__ Ray();
	__host__ __device__ Ray(Point, Point);
	__host__ __device__ Ray(Point, Vector);

	//Returns True if intersection bewteen the ray and the sphere happened at least once
	//float pointers are returns returning t, which can be used to intersection point using getPointFromT
	//intersection point is only valid if t>0, otherwise conllision happened behind ray
	__host__ __device__ bool intersects(const Sphere&, float*, float*) const;
	__host__ __device__ Point getPointFromT(float t) const;

	Point o;
	Vector d;

private:
};
