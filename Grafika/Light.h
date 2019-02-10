#pragma once

#include "Sphere.h"

class Light : public Sphere
{
public:
	DEVICE_PREFIX Light() { shape = LIGHT; };
	DEVICE_PREFIX Light(Sphere s, float i) : Sphere(s){
		intenisty = i;
		shape = LIGHT;
	}

	float intenisty;

	DEVICE_PREFIX inline float R() { return color.r * color.r * intenisty; }
	DEVICE_PREFIX inline float G() { return color.g * color.g * intenisty; }
	DEVICE_PREFIX inline float B() { return color.b * color.b * intenisty; }

private:

};
