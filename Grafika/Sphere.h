#pragma once

#include "Vector.h"

class Sphere
{
public:
	DEVICE_PREFIX Sphere();
	DEVICE_PREFIX Sphere(Point C, float R);
	DEVICE_PREFIX Vector Normal(Point&) const;

	Point c;
	float r;

private:

};