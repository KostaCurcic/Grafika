#pragma once

#include "Vector.h"

enum EShape
{
	TRIANGLE,
	SPHERE
};

class GraphicsObject
{
public:

	DEVICE_PREFIX GraphicsObject() {
		mirror = false;
		r = g = b = 100;
	}

	EShape shape;
	char r, g, b;
	bool mirror;
	//virtual DEVICE_PREFIX Vector Normal(Point&) const = 0;

private:

};
