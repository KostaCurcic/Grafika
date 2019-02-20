#pragma once

#include "Vector.h"
#include "Color.h"
#include "Material.h"

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
	}

	EShape shape;
	Material mat;
	//virtual DEVICE_PREFIX Vector Normal(Point&) const = 0;

private:

};
