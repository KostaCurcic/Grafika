#include "Color.h"

#include <math.h>

DEVICE_PREFIX ColorReal ColorReal::operator+(const ColorReal &c) const
{
	return ColorReal(r + c.r, g + c.g, b + c.b);
}

DEVICE_PREFIX ColorReal ColorReal::operator-(const ColorReal &c) const
{
	return ColorReal(r - c.r, g - c.g, b - c.b);
}

DEVICE_PREFIX ColorReal ColorReal::operator*(const float &f) const
{
	return ColorReal(r * f, g * f, b * f);
}

DEVICE_PREFIX ColorReal ColorReal::operator/(const float &f) const
{
	return ColorReal(r / f, g / f, b / f);
}

DEVICE_PREFIX ColorReal ColorReal::operator*(const ColorReal &c) const
{
	return ColorReal(r * c.r, g * c.g, b * c.b);
}

DEVICE_PREFIX ColorReal ColorReal::operator/(const ColorReal &c) const
{
	return ColorReal(r / c.r, g / c.g, b / c.b);
}

DEVICE_PREFIX ColorReal & ColorReal::operator*=(const float &f)
{
	r *= f;
	g *= f;
	b *= f;
	return *this;
}

DEVICE_PREFIX ColorReal & ColorReal::operator/=(const float &f)
{
	r /= f;
	g /= f;
	b /= f;
	return *this;
}

DEVICE_PREFIX ColorReal & ColorReal::operator*=(const ColorReal &c)
{
	r *= c.r;
	g *= c.g;
	b *= c.b;
	return *this;
}

DEVICE_PREFIX ColorReal & ColorReal::operator/=(const ColorReal &c)
{
	r /= c.r;
	g /= c.g;
	b /= c.b;
	return *this;
}

DEVICE_PREFIX ColorReal & ColorReal::operator+=(const ColorReal &c)
{
	r += c.r;
	g += c.g;
	b += c.b;
	return *this;
}

DEVICE_PREFIX ColorReal & ColorReal::operator-=(const ColorReal &c)
{
	r -= c.r;
	g -= c.g;
	b -= c.b;
	return *this;
}

DEVICE_PREFIX Color ColorReal::getPixColor(float gamma, float exp)
{
	ColorReal ret(powf(r * 255.0f * exp, 1 / gamma),
			      powf(g * 255.0f * exp, 1 / gamma),
			      powf(b * 255.0f * exp, 1 / gamma));

	return Color(ret.r > 255 ? 255 : ret.r,
				 ret.g > 255 ? 255 : ret.g,
				 ret.b > 255 ? 255 : ret.b);
}

DEVICE_PREFIX Color ColorReal::getPixColor()
{
	ColorReal ret(r * 255.0f,
				  g * 255.0f,
				  b * 255.0f);

	return Color(ret.r > 255 ? 255 : ret.r,
				 ret.g > 255 ? 255 : ret.g,
				 ret.b > 255 ? 255 : ret.b);
}

DEVICE_PREFIX ColorReal ColorReal::getColorIntesity(float gamma)
{
	return ColorReal(powf(r, gamma),
					 powf(g, gamma),
					 powf(b, gamma));
}

DEVICE_PREFIX ColorReal ColorReal::fromIntensity(float gamma)
{
	return ColorReal(powf(r, 1 / gamma),
					 powf(g, 1 / gamma),
					 powf(b, 1 / gamma));
}

/*DEVICE_PREFIX ColorReal Color::getRefMultiplier(float gamma) const
{
	float bMax = powf(255.0f, gamma);
	return ColorReal(powf(r, gamma) / bMax,
				     powf(g, gamma) / bMax,
				     powf(b, gamma) / bMax);
}

DEVICE_PREFIX ColorReal Color::getColorIntensity(float gamma) const
{
	return ColorReal(powf(r, gamma),
				     powf(g, gamma),
				     powf(b, gamma));
}*/

DEVICE_PREFIX ColorReal Color::toColorReal() const
{
	return ColorReal(r / 255.0f,
					 g / 255.0f,
					 b / 255.0f);
}
