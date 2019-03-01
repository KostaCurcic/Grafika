#include "Ray.h"

#include <math.h>

DEVICE_PREFIX Ray::Ray()
{
	o = Point();
	d = Vector();
}

DEVICE_PREFIX Ray::Ray(Point p1, Point p2)
{
	o = p1;

	Vector vec = p2 - p1;
	d = vec.Normalize();
}

DEVICE_PREFIX Ray::Ray(Point p1, Vector vec)
{
	o = p1;
	d = vec.Normalize();
}

DEVICE_PREFIX bool Ray::intersects(const Sphere &s, ColorReal* col, float *c1, float *c2) const
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
		if (s.cut) {
			if (t1 > 0 && (s.cutVector * (getPointFromT(t1) - s.cutPoint) < 0)) {
				t1 = -1;
			}
			if (t2 > 0 && (s.cutVector * (getPointFromT(t2) - s.cutPoint) < 0)) {
				t2 = -1;
			}
		}
		if (t1 < 0 && t2 < 0) {
			return false;
		}
		else if ((t1 < t2 && t1 > 0) || t2 < 0) {
			if (c1 != nullptr) *c1 = t1;
			if (c2 != nullptr) *c2 = t2;
		}
		else {
			if (c1 != nullptr) *c1 = t2;
			if (c2 != nullptr) *c2 = t1;
		}
		if(col != nullptr) *col = s.mat.getColor(0, 0);
		return true;
	}
}

DEVICE_PREFIX bool Ray::intersects(const Triangle &tr, ColorReal* col, float *t) const
{
	if (tr.n * d == 0) return false;

	float c1, c2, c3, tt;

	tt = - (tr.n * (o - tr.v0)) / (tr.n * d);

	if (tt < 0) return false;

	Point colp = o + d * tt;

	float weigths[2];

	if(tr.interpolatePoint(colp, (float*)&(tr.t0), (float*)&(tr.t1), (float*)&(tr.t2), weigths, 2)){
		if (t != nullptr) {
			*t = tt;
		}
		if (col != nullptr) {
			*col = tr.mat.getColor(weigths[0], weigths[1]);
		}
		return true;
	}
	return false;
}

DEVICE_PREFIX bool Ray::intersects(const Light &l, ColorReal* col, float *t) const
{
	return intersects((Sphere)l, col, t);
}

DEVICE_PREFIX bool Ray::intersects(const GraphicsObject *g, ColorReal* col, float *t) const
{
	switch (g->shape)
	{
	case TRIANGLE:
		return intersects(*((Triangle*)g), col, t);
	case SPHERE:
	case LIGHT:
		return intersects(*((Sphere*)g), col, t);
	}
}

DEVICE_PREFIX Point Ray::getPointFromT(float t) const
{
	return o + d * t;
}
