#pragma once

#include "Vector.h"
#include "Sphere.h"


class Ray
{
public:
	Ray();
	Ray(Point, Point);
	Ray(Point, Vector);

	bool intersects(const Sphere&) const;

	Point o;
	Vector d;

private:
};
