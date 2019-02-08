#include "Ray.h"

#ifndef CUDA

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
		else if (t1 < t2 && t1 > 0) {
			if (c1 != nullptr) *c1 = t1;
			if (c2 != nullptr) *c2 = t2;
		}
		else {
			if (c1 != nullptr) *c1 = t2;
			if (c2 != nullptr) *c2 = t1;
			return true;
		}
	}
}

bool Ray::intersects(const Triangle &tr, float *t) const
{
	if (tr.n * d == 0) return false;

	float c1, c2, c3, tt;

	tt = - (tr.n * (o - tr.v0)) / (tr.n * d);

	if (tt < 0) return false;

	Point col = o + d * tt;

	c1 = (tr.e0 % (col - tr.v0)) * tr.n;
	c2 = (tr.e1 % (col - tr.v1)) * tr.n;
	c3 = (tr.e2 % (col - tr.v2)) * tr.n;

	if ((c1 <= 0.0001 && c2 <= 0.0001 && c3 <= 0.0001) || (c1 >= -0.0001 && c2 >= -0.0001 && c3 >= -0.0001)) {
		if (t != nullptr) {
			*t = tt;
		}
		return true;
	}
	return false;
}

Point Ray::getPointFromT(float t) const
{
	return o + d * t;
}
#endif
