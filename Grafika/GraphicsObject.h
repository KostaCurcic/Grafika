#pragma once

#include "Vector.h"
#include "Color.h"

enum EShape
{
	TRIANGLE,
	SPHERE,
	LIGHT
};

class GraphicsObject
{
public:

	DEVICE_PREFIX GraphicsObject() {
		mirror = false;
		color = ColorReal(1, 1, 1);
	}

	EShape shape;
	ColorReal color;
	bool mirror;
	//virtual DEVICE_PREFIX Vector Normal(Point&) const = 0;

private:

};
