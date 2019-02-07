#include "Sphere.h"

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
