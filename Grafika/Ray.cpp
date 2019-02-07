#include "Ray.h"
#include <math.h>

Ray::Ray()
{
	o = Point();
	d = Vector();
}

Ray::Ray(Point p1, Point p2)
{
	o = p1;

	Vector vec = p2 - p1;
	d = vec.Normalize();
}

Ray::Ray(Point p1, Vector vec)
{
	o = p1;
	d = vec.Normalize();
}

bool Ray::intersects(const Sphere &s) const
{
	float b = 2 * (d * (o - s.c));
	float c = powf(((Vector)(o - s.c)).Length(), 2) - s.r * s.r;

	if (b * b - 4 * c > 0) {
		return true;
	}
	return false;
}
