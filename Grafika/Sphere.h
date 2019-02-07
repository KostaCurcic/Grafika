#pragma once

#include "Vector.h"

class Sphere
{
public:
	Sphere();
	Sphere(Point C, float R);
	Vector Normal(Point&) const;

	Point c;
	float r;

private:

};