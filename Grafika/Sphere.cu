#include "Sphere.h"

#ifdef CUDA

DEVICE_PREFIX Sphere::Sphere()
{
	c = Point(0, 0, 0);
	r = 0;
}

DEVICE_PREFIX Sphere::Sphere(Point C, float R)
{
	c = C;
	r = R;
}

DEVICE_PREFIX Vector Sphere::Normal(Point &p) const
{
	return ((Vector)(p - c)).Normalize();
}
#endif
