#include "Point.h"

#ifdef CUDA

DEVICE_PREFIX Point::Point()
{
	x = y = z = 0;
}

DEVICE_PREFIX Point::Point(float X, float Y, float Z)
{
	x = X;
	y = Y;
	z = Z;
}

DEVICE_PREFIX Point Point::operator+(const Point &p) const
{
	return Point(x + p.x, y + p.y, z + p.z);
}

DEVICE_PREFIX Point Point::operator-(const Point &p) const
{
	return Point(x - p.x, y - p.y, z - p.z);
}
#endif