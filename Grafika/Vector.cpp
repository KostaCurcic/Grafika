#include "Vector.h"

#ifndef CUDA

#include <math.h>

Vector::Vector(const Point &p)
{
	x = p.x;
	y = p.y;
	z = p.z;
}

Vector& Vector::Unzero()
{
	if (x == 0.0f) {
		x = 0.00001f;
	}
	if (y == 0.0f) {
		y = 0.00001f;
	}
	if (z == 0.0f) {
		z = 0.00001f;
	}
	return *this;
}

Vector & Vector::Normalize()
{
	float sum = Length();

	x /= sum;
	y /= sum;
	z /= sum;

	return *this;
}

float Vector::Length() const
{
	return sqrtf(x * x + y * y + z * z);
}

float Vector::operator*(const Vector &v) const
{
	return x * v.x + y * v.y + z * v.z;
}

Vector Vector::operator*(const float s) const
{
	return Vector(x * s, y * s, z * s);
}

Vector Vector::operator%(const Vector &p) const
{
	return Vector(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x);
}

#endif
