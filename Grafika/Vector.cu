#include "Vector.cuh"
#include <math.h>

__host__ __device__ Vector::Vector(const Point &p)
{
	x = p.x;
	y = p.y;
	z = p.z;
}

__host__ __device__ Vector& Vector::Unzero()
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

__host__ __device__ Vector & Vector::Normalize()
{
	float sum = Length();

	x /= sum;
	y /= sum;
	z /= sum;

	return *this;
}

__host__ __device__ float Vector::Length() const
{
	return sqrtf(x * x + y * y + z * z);
}

__host__ __device__ float Vector::operator*(const Vector &v) const
{
	return x * v.x + y * v.y + z * v.z;
}

__host__ __device__ Vector Vector::operator*(const float s) const
{
	return Vector(x * s, y * s, z * s);
}
