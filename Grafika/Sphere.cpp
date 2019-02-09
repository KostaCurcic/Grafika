#include "Sphere.h"

#ifndef CUDA

DEVICE_PREFIX Sphere::Sphere()
{
	c = Point(0, 0, 0);
	r = 0;
	shape = SPHERE;
}

DEVICE_PREFIX Sphere::Sphere(Point C, float R)
{
	c = C;
	r = R;
	shape = SPHERE;
}

Vector Sphere::Normal(Point &p) const
{
	return ((Vector)(p - c)).Normalize();
}
#endif

