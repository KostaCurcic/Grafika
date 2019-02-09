#include "Drawing.h"
#include "Ray.h"

#ifndef CUDA

#include <math.h>
#include <Windows.h>

//#define NONRT

#define SPHC 2
#define LIGHTS 2
#define TRIS 3

#define THRCOUNT 4

Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];
Sphere lights[LIGHTS];
Triangle triangles[TRIS];
float angle = 0;
char *imgptr;
int signal = 0;

void InitFrame()
{
	spheres[0] = Sphere(Point(sinf(angle) * 3, -1, 10 + cosf(angle) * 3), 1);
	//spheres[0].mirror = true;

	spheres[1] = Sphere(Point(5, -1, 5), 1);
	spheres[1].color.r = 50;
	spheres[1].color.g = 200;
	spheres[1].color.b = 100;

	lights[0] = Sphere(Point(2, 2, 10), 0.2);
	lights[1] = Sphere(Point(-7, 0, 6), 0.5);
	triangles[0] = Triangle(Point(10, -2, 0), Point(-10, -2, 0), Point(10, -2, 20));
	triangles[1] = Triangle(Point(-10, -2, 0), Point(-10, -2, 20), Point(10, -2, 20));

	triangles[2] = Triangle(Point(-6, 2, 6), Point(-5, -2, 8), Point(-5, -5, 4));
	//triangles[2].mirror = true;
	//triangles[2].color.r = 240;

	//angle += 0.01;
}

#ifdef NONRT

float *realImg;
int iteration[THRCOUNT];

float findColPoint(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj) {

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

	for (int i = 0; i < LIGHTS; i++) {
		if (ray.intersects(lights[i], &t1, nullptr)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = lights[i].Normal(*colPoint);
				*colObj = lights + i;
				mirror = lights[i].mirror;
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

void drawPixelR(float x, float y, float *pix) {
	Point pixelPoint(x, y, 0);

	Point camera = Point(0, 0, -2.0f);
	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(camera, pixelPoint);

	float light;
	float ra;

	Point colPoint;

	int bounceCount = 5;

	for (bounceCount = 5; bounceCount > 0; bounceCount--) {
		if (!findColPoint(ray, &colPoint, &normal, &obj)) {
			pix[0] += 1;
			pix[1] += 5;
			pix[2] += 10;
			return;
		}
		if (obj >= lights && obj < lights + LIGHTS) {
			if (obj == lights) {
				pix[0] += 5000;
				pix[1] += 1500;
				pix[2] += 400;
			}
			else {
				pix[0] += 1000;
				pix[1] += 3000;
				pix[2] += 200;
			}
			return;
		}
		ray.o = colPoint;
		do {
			ray.d.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f;
			ray.d.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f;
			ray.d.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f;
			ray.d.Normalize();
		} while (ray.d * normal <= 0);
	}

	/*if (findColPoint(ray, &colPoint, &normal, &obj)) {
		light = pointLit(colPoint, normal, obj);
		pix[0] = obj->color.r * light;
		pix[1] = obj->color.g * light;
		pix[2] = obj->color.b * light;
	}
	else {
		pix[0] = 40;
		pix[1] = 120;
		pix[2] = 240;

	}*/
}

DWORD WINAPI ThreadFunc(void* data) {
	while (true) {
		//WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);

		int size = YRES / THRCOUNT;
		float rc, gc, bc;

		int limit = (int)data * size + size;
		for (int t = 0; t < 200000; t++) {
			int i = rand() % YRES;
			int j = rand() % XRES;
			drawPixelR(j * 2.0f / YRES - XRES / (float)YRES + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / YRES
				, i * 2.0 / YRES - 1.0 + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / XRES, realImg + (i * XRES + j) * 3);
			rc = realImg[(i * XRES + j) * 3] / iteration[(int)data] * 5;
			gc = realImg[(i * XRES + j) * 3 + 1] / iteration[(int)data] * 5;
			bc = realImg[(i * XRES + j) * 3 + 2] / iteration[(int)data] * 5;

			if (rc > 255) rc = 255;
			if (gc > 255) gc = 255;
			if (bc > 255) bc = 255;

			imgptr[(i * XRES + j) * 3] = rc;
			imgptr[(i * XRES + j) * 3 + 1] = gc;
			imgptr[(i * XRES + j) * 3 + 2] = bc;

			/*imgptr[(i * XRES + j) * 3] = realImg[(i * XRES + j) * 3] / iteration;
			imgptr[(i * XRES + j) * 3 + 1] = realImg[(i * XRES + j) * 3 + 1] / iteration;
			imgptr[(i * XRES + j) * 3 + 2] = realImg[(i * XRES + j) * 3 + 2] / iteration;*/

		}
		iteration[(int)data]++;

		//signal--;
		//WakeByAddressSingle(&signal);
	}
	return 0;
}

void DrawFrame() {};

void InitDrawing(char * ptr)
{
	InitFrame();
	imgptr = ptr;
	realImg = (float*)malloc(XRES * YRES * 3 * sizeof(float));
	ZeroMemory(realImg, XRES * YRES * 3 * sizeof(float));
	for (int i = 0; i < THRCOUNT; i++) {
		iteration[i] = 1;
		HANDLE thread = CreateThread(NULL, 0, ThreadFunc, (void*)i, 0, NULL);
	}
}

#else

float pointLit(Point &p, Vector n, GraphicsObject* self) {
	Ray ray;
	float lit = 0, t;
	bool col;
	for (int i = 0; i < LIGHTS; i++) {
		ray = Ray(p, lights[i].c);
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
		pix[0] = obj->color.r * light;
		pix[1] = obj->color.g * light;
		pix[2] = obj->color.b * light;
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

#endif
