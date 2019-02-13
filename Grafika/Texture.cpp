#include "Texture.h"
#include <stdio.h>
#include <Windows.h>

#include "Point.h"

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
		#ifdef CUDA
			cudaFree(data);
		#else
			free(data);
		#endif // CUDA
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

	#ifdef CUDA
		unsigned char* devData;
		cudaMalloc(&devData, size);

		cudaMemcpy(devData, data, size, cudaMemcpyHostToDevice);

		free(data);
		data = devData;
	#endif
}

DEVICE_PREFIX Color Texture::getColor(float x, float y, bool bilinear)
{
	float xc = (x * width);
	float yc = (y * height);
	Color ret;
	if (bilinear) bilinearTexGet(xc, yc, &ret);
	else nearestTexGet(xc, yc, &ret);
	return ret;
}

DEVICE_PREFIX void Texture::nearestTexGet(float x, float y, Color *ret) {
	*ret = Color(data[(((int)y) * width + (int)x) * 3], data[(((int)y) * width + (int)x) * 3 + 1], data[(((int)y) * width + (int)x) * 3 + 2]);
}
DEVICE_PREFIX void Texture::bilinearTexGet(float x, float y, Color *ret) {
	int r, g, b;
	int xc = (int)x, yc = (int)y;

	if (xc >= width - 2 || yc >= height - 2) {
		nearestTexGet(x, y, ret);
		return;
	}

	float xOff = x - xc, yOff = y - yc;

	xc *= 3;
	yc *= 3;

	r = (data[yc * width + xc] * (1 - xOff) + data[yc * width + xc + 3] * xOff)	* (1 - yOff) + (data[(yc + 3) * width + xc] * (1 - xOff) + data[(yc + 3) * width + xc + 3] * xOff) * yOff;
	g = (data[yc * width + xc + 1] * (1 - xOff) + data[yc * width + xc + 4] * xOff)	* (1 - yOff) + (data[(yc + 3) * width + xc + 1] * (1 - xOff) + data[(yc + 3) * width + xc + 4] * xOff) * yOff;
	b = (data[yc * width + xc + 2] * (1 - xOff) + data[yc * width + xc + 5] * xOff)	* (1 - yOff) + (data[(yc + 3) * width + xc + 2] * (1 - xOff) + data[(yc + 3) * width + xc + 5] * xOff) * yOff;

	*ret = Color(r, g, b);
}
