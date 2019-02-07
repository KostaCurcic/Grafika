#pragma once

class Point
{
public:
	Point();
	Point(float, float, float);

	Point operator+(const Point&) const;
	Point operator-(const Point&) const;

	float x, y, z;

private:
};
