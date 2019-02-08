#pragma once

#include "Vector.h"

class Triangle
{
public:
	DEVICE_PREFIX Triangle();
	DEVICE_PREFIX Triangle(Point, Point, Point);

	/*DEVICE_PREFIX Point V0() const;
	DEVICE_PREFIX Point V1() const;
	DEVICE_PREFIX Point V2() const;
	DEVICE_PREFIX Vector N() const;

	DEVICE_PREFIX void setV0(Point);
	DEVICE_PREFIX void setV1(Point);
	DEVICE_PREFIX void setV2(Point);*/

	Point v0, v1, v2;
	Vector n, e0, e1, e2;

private:

	DEVICE_PREFIX void calcVectors();

};
