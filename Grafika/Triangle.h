#pragma once

#include "Vector.h"
#include "GraphicsObject.h"

class Triangle : public GraphicsObject
{
public:
	DEVICE_PREFIX Triangle();
	DEVICE_PREFIX Triangle(Point, Point, Point);
	DEVICE_PREFIX Vector Normal(Point&) const;

	DEVICE_PREFIX bool interpolatePoint(const Point &p, float *v0val, float *v1val, float *v2val, float *pval, int n) const;

	/*DEVICE_PREFIX Point V0() const;
	DEVICE_PREFIX Point V1() const;
	DEVICE_PREFIX Point V2() const;
	DEVICE_PREFIX Vector N() const;

	DEVICE_PREFIX void setV0(Point);
	DEVICE_PREFIX void setV1(Point);
	DEVICE_PREFIX void setV2(Point);*/

	Point v0, v1, v2;
	Vector n, e0, e1, e2;

	bool textured = false;
	int texIndex = 0;
	Point t0, t1, t2;

private:

	DEVICE_PREFIX void calcVectors();

};
