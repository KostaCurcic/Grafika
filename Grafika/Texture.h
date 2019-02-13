#pragma once

#include "Point.h"
#include "Color.h"

class Texture
{
public:
	Texture();
	Texture(const char *filename);
	~Texture();

	void load(const char *filename);

	DEVICE_PREFIX Color getColor(float x, float y);

	int width, height;

	unsigned char* data;

private:

};
