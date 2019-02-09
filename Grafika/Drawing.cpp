#include "Drawing.h"
#include "Ray.h"

#ifndef CUDA

#include <math.h>
#include <Windows.h>

#define SPHC 1
#define LIGHTS 1
#define TRIS 2

#define THRCOUNT 4

Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];
Point lights[LIGHTS];
Triangle triangles[TRIS];
float angle = 0;
char *imgptr;
int signal = 0;

void InitFrame()
{
	spheres[0] = Sphere(Point(sinf(angle) * 3, -1.5, 10 + cosf(angle) * 3), 1);
	angle += 0.01;
	//spheres[1] = Sphere(Point(0, -1000, 10), 995);
	lights[0] = Point(2, 2, 10);
	triangles[0] = Triangle(Point(10, -2, 0), Point(-10, -2, 0), Point(10, -2, 20));
	triangles[1] = Triangle(Point(-10, -2, 0), Point(-10, -2, 20), Point(10, -2, 20));
	//lights[1] = Point(1000, 0, 0);
}

float pointLit(Point &p, Vector n, GraphicsObject* self) {
	Ray ray;
	float lit = 0, t;
	bool col;
	for (int i = 0; i < LIGHTS; i++) {
		ray = Ray(p, lights[i]);
		if (n * ray.d > 0) {
			col = false;
			for (int j = 0; j < SPHC; j++) {
				if (spheres + j != self && ray.intersects(spheres[j], &t) && t > 0.001) {
					col = true;
					break;
				}
			}
			if (!col) {
				for (int j = 0; j < TRIS; j++) {
					if (triangles + j != self && ray.intersects(triangles[j], &t) && t > 0.001) {
						col = true;
						break;
					}
				}
			}
			if (!col) {
				lit += n * ray.d;
			}
		}
	}
	return lit;
}

bool findColPoint(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj) {

	float t1, nearest = INFINITY;
	bool mirror = false;

	for (int i = 0; i < SPHC; i++) {
		if (ray.intersects(spheres[i], &t1, nullptr)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = spheres[i].Normal(*colPoint);
				*colObj = spheres + i;
				mirror = spheres[i].mirror;
			}
		}
	}

	for (int i = 0; i < TRIS; i++) {
		if (ray.intersects(triangles[i], &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = triangles[i].n;
				*colObj = triangles + i;
				mirror = triangles[i].mirror;
			}
		}
	}

	if (mirror) {
		return findColPoint(Ray(*colPoint, ray.d.Reflect(*colNormal)), colPoint, colNormal, colObj);
	}

	if (nearest < INFINITY) return true;
	return false;
}

void drawPixel(float x, float y, char *pix) {
	Point pixelPoint(x, y, 0);

	Point camera = Point(0, 0, -2.0f);
	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(camera, pixelPoint);

	float light;

	Point colPoint;

	if (findColPoint(ray, &colPoint, &normal, &obj)) {
		light = pointLit(colPoint, normal, obj);
		pix[0] = obj->r * light;
		pix[1] = obj->g * light;
		pix[2] = obj->b * light;
	}
	else {
		pix[0] = 40;
		pix[1] = 120;
		pix[2] = 240;

	}
}

DWORD WINAPI ThreadFunc(void* data) {
	while (true) {
		//WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);

		int size = YRES / THRCOUNT;

		int limit = (int)data * size + size;

		for (int i = (int)data * size; i < limit; i++) {
			for (int j = 0; j < XRES; j++) {
				drawPixel(j * 2.0f / YRES - XRES / (float)YRES, i * 2.0 / YRES - 1.0, imgptr + (i * XRES + j) * 3);
			}
		}
		//signal--;
		//WakeByAddressSingle(&signal);
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
	//signal = THRCOUNT;
	//WakeByAddressAll(&signal);

	/*while (signal > 0) {
		WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);
	}*/
}

#endif
