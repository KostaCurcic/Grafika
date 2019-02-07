#include "Drawing.h"
#include "Ray.h"
#include <math.h>
#include <Windows.h>

#define SPHC 2
#define LIGHTS 1

#define THRCOUNT 5

Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];
Point lights[LIGHTS];
float angle = 0;
char *imgptr;
int signal = 0;

void InitFrame()
{
	spheres[0] = Sphere(Point(sinf(angle) * 3, 0, 10 + cosf(angle) * 3), 1);
	angle += 0.01;
	spheres[1] = Sphere(Point(0, -1000, 10), 995);
	lights[0] = Point(2, 2, 10);
	//spheres[2] = Sphere(Point(2, 2.2, 10), 0.1f);
	//lights[1] = Point(1000, 0, 0);
}

void drawPixel(float x, float y, char *pix) {
	Point pixelPoint(x, y, 0);

	Ray ray = Ray(camera, pixelPoint);
	Ray shadowRay;
	bool collided = false, lit = false, sCollided = false;

	float t1, t2;
	Point colPoint;

	for (int i = 0; i < SPHC; i++) {
		if (ray.intersects(spheres[i], &t1, &t2)) {
			if (t1 > t2 && t2 >= 0) {
				colPoint = ray.getPointFromT(t2);
			}
			else {
				colPoint = ray.getPointFromT(t1);
			}
			lit = false;
			for (int j = 0; j < LIGHTS; j++) {
				shadowRay = Ray(colPoint, lights[j]);
				sCollided = false;
				if (spheres[i].Normal(colPoint) * shadowRay.d > 0) {
					for (int s = 0; s < SPHC; s++) {
						if (s == i) continue;
						if (shadowRay.intersects(spheres[s], nullptr, nullptr)) {
							sCollided = true;
							break;
						}
					}


					if (!sCollided) {
						pix[0] = 50;
						pix[1] = 200;
						pix[2] = 100;
						lit = true;
						break;
					}
				}
			}
			if (!lit) {
				pix[0] = 0;
				pix[1] = 0;
				pix[2] = 0;
			}
			collided = true;
			break;
		}
	}

	if (!collided) {
		pix[0] = 41;
		pix[1] = 119;
		pix[2] = 240;
	}
}

DWORD WINAPI ThreadFunc(void* data) {
	while (true) {
		WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);

		int size = YRES / THRCOUNT;

		int limit = (int)data * size + size;

		for (int i = (int)data * size; i < limit; i++) {
			for (int j = 0; j < XRES; j++) {
				drawPixel(j * 2.0f / YRES - XRES / (float)YRES, i * 2.0 / YRES - 1.0, imgptr + (i * XRES + j) * 3);
			}
		}
		signal--;
		WakeByAddressSingle(&signal);
	}
	return 0;
}

void InitDrawing(char * ptr)
{
	imgptr = ptr;
	for (int i = 0; i < THRCOUNT; i++) {
		HANDLE thread = CreateThread(NULL, 0, ThreadFunc, (void*)i, 0, NULL);
	}
}

void DrawFrame()
{
	InitFrame();
	signal = THRCOUNT;
	WakeByAddressAll(&signal);

	while (signal > 0) {
		WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);
	}
}
