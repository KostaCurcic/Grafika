#pragma once

#include "Point.h"

class Color;

class ColorReal {
public:
	DEVICE_PREFIX ColorReal() {};
	DEVICE_PREFIX ColorReal(float R, float G, float B) {
		r = R;
		g = G;
		b = B;
	}

	DEVICE_PREFIX ColorReal operator+(const ColorReal&) const;
	DEVICE_PREFIX ColorReal operator-(const ColorReal&) const;
	DEVICE_PREFIX ColorReal operator*(const float&) const;
	DEVICE_PREFIX ColorReal operator/(const float&) const;
	DEVICE_PREFIX ColorReal operator*(const ColorReal&) const;
	DEVICE_PREFIX ColorReal operator/(const ColorReal&) const;
	DEVICE_PREFIX ColorReal& operator*=(const float&);
	DEVICE_PREFIX ColorReal& operator/=(const float&);
	DEVICE_PREFIX ColorReal& operator*=(const ColorReal&);
	DEVICE_PREFIX ColorReal& operator/=(const ColorReal&);
	DEVICE_PREFIX ColorReal& operator+=(const ColorReal&);
	DEVICE_PREFIX ColorReal& operator-=(const ColorReal&);

	DEVICE_PREFIX Color getPixColor(float gamma, float exp);
	DEVICE_PREFIX Color getPixColor();

	DEVICE_PREFIX ColorReal getColorIntesity(float gamma);
	DEVICE_PREFIX ColorReal fromIntensity(float gamma);

	float r, g, b;
};

class Color
{
public:
	DEVICE_PREFIX Color() {};
	DEVICE_PREFIX Color(char R, char G, char B) {
		r = R;
		g = G;
		b = B;
	}

	//DEVICE_PREFIX ColorReal getRefMultiplier(float gamma) const;
	//DEVICE_PREFIX ColorReal getColorIntensity(float gamma) const;

	DEVICE_PREFIX ColorReal toColorReal() const;

	unsigned char r, g, b;

private:

};