#include "Point.h"

#ifndef CUDA

Point::Point()
{
	x = y = z = 0;
}

Point::Point(float X, float Y, float Z)
{
	x = X;
	y = Y;
	z = Z;
}

Point Point::operator+(const Point &p) const
{
	return Point(x + p.x, y + p.y, z + p.z);
}

Point Point::operator-(const Point &p) const
{
	return Point(x - p.x, y - p.y, z - p.z);
}
#endif