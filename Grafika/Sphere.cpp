#include "Sphere.h"

#ifndef CUDA

Sphere::Sphere()
{
	c = Point(0, 0, 0);
	r = 0;
}

Sphere::Sphere(Point C, float R)
{
	c = C;
	r = R;
}

Vector Sphere::Normal(Point &p) const
{
	return ((Vector)(p - c)).Normalize();
}
#endif

