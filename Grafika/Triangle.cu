#include "Triangle.h"

#ifdef CUDA

DEVICE_PREFIX Triangle::Triangle() {}

DEVICE_PREFIX Triangle::Triangle(Point ver1, Point ver2, Point ver3)
{
	v0 = ver1;
	v1 = ver2;
	v2 = ver3;
	calcVectors();
}

/*

DEVICE_PREFIX Point Triangle::V0() const
{
	return v0;
}

DEVICE_PREFIX Point Triangle::V1() const
{
	return v1;
}

DEVICE_PREFIX Point Triangle::V2() const
{
	return v2;
}

DEVICE_PREFIX Vector Triangle::N() const
{
	return n;
}

DEVICE_PREFIX void Triangle::setV0(Point v)
{
	v0 = v;
	calcVectors();
}

DEVICE_PREFIX void Triangle::setV1(Point v)
{
	v1 = v;
	calcVectors();
}

DEVICE_PREFIX void Triangle::setV2(Point v)
{
	v2 = v;
	calcVectors();
}*/

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

#endif