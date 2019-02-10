#pragma once

#include "Point.h"

class Color
{
public:
	DEVICE_PREFIX Color() {};
	DEVICE_PREFIX Color(char R, char G, char B) {
		r = R;
		g = G;
		b = B;
	}

	unsigned char r, g, b;

private:

};