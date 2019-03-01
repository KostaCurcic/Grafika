#pragma once

#include "Vector.h"
#include "GraphicsObject.h"

class Sphere : public GraphicsObject
{
public:
	DEVICE_PREFIX Sphere();
	DEVICE_PREFIX Sphere(Point C, float R);
	DEVICE_PREFIX Vector Normal(const Point&) const;

	Point c;
	float r;

	bool cut = false;
	Point cutPoint;
	Vector cutVector;

private:

};