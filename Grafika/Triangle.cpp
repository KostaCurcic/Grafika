#include "Triangle.h"

#ifndef CUDA

Triangle::Triangle() {}

Triangle::Triangle(Point ver1, Point ver2, Point ver3)
{
	v0 = ver1;
	v1 = ver2;
	v2 = ver3;
	calcVectors();
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

void Triangle::calcVectors()
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