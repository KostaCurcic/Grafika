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

private:

};
