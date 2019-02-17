#pragma once

#include "Vector.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Light.h"


class Ray
{
public:
	DEVICE_PREFIX Ray();
	DEVICE_PREFIX Ray(Point, Point);
	DEVICE_PREFIX Ray(Point, Vector);

	//Returns True if intersection bewteen the ray and the sphere happened at least once
	//float pointers are returns returning t, which can be used to intersection point using getPointFromT
	//first returned t point is always the closer valid one
	DEVICE_PREFIX bool intersects(const Sphere&, ColorReal*, float*, float* = nullptr) const;

	DEVICE_PREFIX bool intersects(const Triangle&, ColorReal*, float*) const;

	DEVICE_PREFIX bool intersects(const Light&, ColorReal*, float*) const;

	DEVICE_PREFIX bool intersects(const GraphicsObject*, ColorReal*, float*) const;

	DEVICE_PREFIX Point getPointFromT(float t) const;

	Point o;
	Vector d;

private:
};
