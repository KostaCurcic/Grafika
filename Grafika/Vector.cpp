#include "Vector.h"
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
