#pragma once

#include "Point.h"
#include "Color.h"

class Texture
{
public:
	Texture();
	Texture(const char *filename);
	void Unload();

	void load(const char *filename);

	DEVICE_PREFIX ColorReal getColor(float x, float y, bool bilinear = true);

	int width, height;

	float* data;

private:
	DEVICE_PREFIX void nearestTexGet(float, float, ColorReal*);
	DEVICE_PREFIX void bilinearTexGet(float, float, ColorReal*);
};
