#include "Vector.h"

#ifdef CUDA

#include <math.h>

DEVICE_PREFIX Vector::Vector(const Point &p)
{
	x = p.x;
	y = p.y;
	z = p.z;
}

DEVICE_PREFIX Vector& Vector::Unzero()
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

DEVICE_PREFIX Vector & Vector::Normalize()
{
	float sum = Length();

	x /= sum;
	y /= sum;
	z /= sum;

	return *this;
}

DEVICE_PREFIX float Vector::Length() const
{
	return sqrtf(x * x + y * y + z * z);
}

DEVICE_PREFIX float Vector::operator*(const Vector &v) const
{
	return x * v.x + y * v.y + z * v.z;
}

DEVICE_PREFIX Vector Vector::operator*(const float s) const
{
	return Vector(x * s, y * s, z * s);
}

DEVICE_PREFIX Vector Vector::operator%(const Vector &p) const
{
	return Vector(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x);
}
#endif
