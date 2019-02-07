#pragma once

#include "Point.h"

class Vector : public Point
{
public:
	Vector(const Point&);
	Vector() {};
	Vector(float X, float Y, float Z) : Point(X, Y, Z) {};
	Vector& Unzero();
	Vector& Normalize();
	float Length() const;
	float operator*(const Vector&) const;

private:

};
