#pragma once

#include "Vector.h"
#include "Sphere.h"


class Ray
{
public:
	Ray();
	Ray(Point, Point);
	Ray(Point, Vector);

	//Returns True if intersection bewteen the ray and the sphere happened at least once
	//float pointers are returns returning t, which can be used to intersection point using getPointFromT
	//intersection point is only valid if t>0, otherwise conllision happened behind ray
	bool intersects(const Sphere&, float*, float*) const;
	Point getPointFromT(float t) const;

	Point o;
	Vector d;

private:
};
