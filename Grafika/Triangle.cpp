#include "Triangle.h"

DEVICE_PREFIX Triangle::Triangle()
{
	shape = TRIANGLE;
}

DEVICE_PREFIX Triangle::Triangle(Point ver1, Point ver2, Point ver3)
{
	v0 = ver1;
	v1 = ver2;
	v2 = ver3;
	calcVectors();
	shape = TRIANGLE;
}

/*

Point Triangle::V0() const
{
	return v0;
}

Point Triangle::V1() const
{
	return v1;
}

Point Triangle::V2() const
{
	return v2;
}

Vector Triangle::N() const
{
	return n;
}

void Triangle::setV0(Point v)
{
	v0 = v;
	calcVectors();
}

void Triangle::setV1(Point v)
{
	v1 = v;
	calcVectors();
}

void Triangle::setV2(Point v)
{
	v2 = v;
	calcVectors();
}*/

DEVICE_PREFIX bool Triangle::interpolatePoint(const Point & p, float * v0val, float * v1val, float * v2val, float * pval, int n) const
{
	float surface = ((Vector)(v1 - v0) % (v2 - v0)).Length() / 2;
	float v0w = (((Vector)(p - v1) % (p - v2)).Length() / 2) / surface;
	float v1w = (((Vector)(p - v2) % (p - v0)).Length() / 2) / surface;
	float v2w = (((Vector)(p - v1) % (p - v0)).Length() / 2) / surface;
	for (int i = 0; i < n; i++) {
		pval[i] = v0val[i] * v0w + v1val[i] * v1w + v2val[i] * v2w;
	}
	if (v0w + v1w + v2w < 1.001f) return true;
	return false;
}

DEVICE_PREFIX void Triangle::calcVectors()
{
	e0 = v1 - v0;
	e1 = v2 - v1;
	e2 = v0 - v2;

	e0.Normalize();
	e1.Normalize();
	e2.Normalize();

	n = (e0 % e1).Normalize();
}

DEVICE_PREFIX Vector Triangle::Normal(Point &) const
{
	return n;
}