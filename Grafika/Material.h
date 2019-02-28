#pragma once

#include "Color.h"
#include "Texture.h"
#include <string>

using namespace std;

class Material
{
public:
	DEVICE_PREFIX Material();
	DEVICE_PREFIX Material(ColorReal &);
	DEVICE_PREFIX Material(Texture &);
	DEVICE_PREFIX Material(const char *path);

	DEVICE_PREFIX ColorReal getColor(float x, float y) const;

	bool mirror = false;

	bool transparent = false;
	float refIndex = 1.0f;

	char name[100] = { 0 };
	Texture texture;
	ColorReal color;

private:

};