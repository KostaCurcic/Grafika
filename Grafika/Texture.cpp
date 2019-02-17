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

void Texture::Unload()
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
	data = (float*)malloc(size * sizeof(float)); // allocate 3 bytes per pixel
	unsigned char temp[3];

	for (i = 0; i < size; i += 3)
	{
		fread(temp, sizeof(unsigned char), 3, f);
		data[i] = temp[2] / 255.0f;
		data[i + 1] = temp[1] / 255.0f;
		data[i + 2] = temp[0] / 255.0f;
	}
	fclose(f);

	#ifdef CUDA
		float* devData;
		cudaMalloc(&devData, size * sizeof(float));

		cudaMemcpy(devData, data, size * sizeof(float), cudaMemcpyHostToDevice);

		free(data);
		data = devData;
	#endif
}

DEVICE_PREFIX ColorReal Texture::getColor(float x, float y, bool bilinear)
{
	float xc = (x * width);
	float yc = (y * height);

	while (xc >= (float)width) {
		xc -= width;
	}
	while (xc < 0) {
		xc += width;
	}
	while (yc >= (float)height) {
		yc -= height;
	}
	while (yc < 0) {
		yc += height;
	}

	ColorReal ret;
	if (bilinear) bilinearTexGet(xc, yc, &ret);
	else nearestTexGet(xc, yc, &ret);
	return ret;
}

DEVICE_PREFIX void Texture::nearestTexGet(float x, float y, ColorReal *ret) {
	if(x < width && y < height && x >= 0 && y >= 0)
		*ret = ColorReal(data[(((int)y) * width + (int)x) * 3], data[(((int)y) * width + (int)x) * 3 + 1], data[(((int)y) * width + (int)x) * 3 + 2]);
}
DEVICE_PREFIX void Texture::bilinearTexGet(float x, float y, ColorReal *ret) {
	float r, g, b;
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

	*ret = ColorReal(r, g, b);
}
