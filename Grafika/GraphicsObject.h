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
		color = ColorReal(0.85f, 0.85f, 0.85f);
	}

	EShape shape;
	ColorReal color;
	bool mirror;
	//virtual DEVICE_PREFIX Vector Normal(Point&) const = 0;

private:

};
