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

bool Ray::intersects(const Sphere &s, float *c1, float *c2) const
{
	float b = 2 * (d * (o - s.c));
	float c = powf(((Vector)(o - s.c)).Length(), 2) - s.r * s.r;
	float del = b * b - 4 * c;
	float t1, t2;

	if (del < 0) {
		return false;
	}
	else
	{
		t1 = (-b + sqrt(del)) / 2;
		t2 = (-b - sqrt(del)) / 2;
		if (t1 < 0 && t2 < 0) {
			return false;
		}
		else {
			if (c1 != nullptr) *c1 = t1;
			if (c2 != nullptr) *c2 = t2;
			return true;
		}
	}
}

Point Ray::getPointFromT(float t) const
{
	return o + d * t;
}
