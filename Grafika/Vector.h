#pragma once

#include "Point.h"

class Vector : public Point
{
public:
	DEVICE_PREFIX Vector(const Point&);
	DEVICE_PREFIX Vector() {};
	DEVICE_PREFIX Vector(float X, float Y, float Z) : Point(X, Y, Z) {};
	DEVICE_PREFIX Vector& Unzero();
	DEVICE_PREFIX Vector& Normalize();
	DEVICE_PREFIX float Length() const;
	DEVICE_PREFIX float operator*(const Vector&) const;
	DEVICE_PREFIX Vector operator*(const float) const;
	//CrossProduct
	DEVICE_PREFIX Vector operator%(const Vector&) const;
	DEVICE_PREFIX Vector operator-() const;
	DEVICE_PREFIX Vector operator-(const Vector&) const;
	DEVICE_PREFIX Vector Reflect(const Vector& normal) const;

private:

};
