#include "Material.h"

Material::Material()
{
	color = ColorReal(0.95f, 0.95f, 0.95f);
}

Material::Material(ColorReal &c)
{
	color = c;
}

Material::Material(Texture &t)
{
	color = ColorReal(1, 1, 1);
	texture = t;
}

Material::Material(const char * path)
{
	color = ColorReal(1, 1, 1);
	texture.load(path);
}

DEVICE_PREFIX ColorReal Material::getColor(float x, float y) const
{
	if (texture.width == 0) {
		return color;
	}
	return (color * texture.getColor(x, y));
}
