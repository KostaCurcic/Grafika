#pragma once

#include "Color.h"
#include "Texture.h"

class Material
{
public:
	DEVICE_PREFIX Material();
	DEVICE_PREFIX Material(ColorReal &);
	DEVICE_PREFIX Material(Texture &);
	DEVICE_PREFIX Material(const char *path);

	DEVICE_PREFIX ColorReal getColor(float x, float y) const;

	bool mirror = false;
	Texture texture;
	ColorReal color;

private:

};