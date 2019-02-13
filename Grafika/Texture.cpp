#include "Texture.h"
#include <stdio.h>
#include <Windows.h>

#include "Point.h"

#ifndef CUDA

Texture::Texture()
{
	width = height = 0;
	data = nullptr;
}

Texture::Texture(const char * filename)
{
	load(filename);
}

Texture::~Texture()
{
	if (data != nullptr) {
		free(data);
	}
}

void Texture::load(const char * filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	width = *(int*)&info[18];
	height = *(int*)&info[22];

	int size = 3 * width * height;
	data = (unsigned char*)malloc(size); // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	for (i = 0; i < size; i += 3)
	{
		unsigned char tmp = data[i];
		data[i] = data[i + 2];
		data[i + 2] = tmp;
	}
}

DEVICE_PREFIX Color Texture::getColor(float x, float y)
{
	int xc = (int)(x * width) * 3;
	int yc = (int)(y * height) * 3;
	return Color(data[yc * width + xc], data[yc * width + xc + 1], data[yc * width + xc + 2]);
}

#endif // CUDA
