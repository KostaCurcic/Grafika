#include "Drawing.h"
#include "Ray.h"

#ifndef CUDA

#include <math.h>
#include <Windows.h>

//#define NONRT
#define THRCOUNT 4

float angle = 0;
char *imgptr;
int signal = 0;

SceneData sd;

void InitFrame()
{
	sd.spheres[0] = Sphere(Point(sinf(angle) * 3, -1, 8 + cosf(angle) * 3), 1);
	//spheres[0].mirror = true;

	sd.spheres[1] = Sphere(Point(5, -1, 5), 1);
	sd.spheres[1].color.r = 50;
	sd.spheres[1].color.g = 200;
	sd.spheres[1].color.b = 100;

	sd.lights[0] = Light(Sphere(Point(-100, 100, 10), 10), 1.0f);
	sd.lights[0].color.r = 239;
	sd.lights[0].color.g = 163;
	sd.lights[0].color.b = 56;


	//lights[1] = Sphere(Point(-7, 0, 6), 0.5);
	sd.triangles[0] = Triangle(Point(10, -2, 0), Point(-10, -2, 0), Point(10, -2, 20));
	sd.triangles[1] = Triangle(Point(-10, -2, 0), Point(-10, -2, 20), Point(10, -2, 20));

	/*triangles[2] = Triangle(Point(-4, 2, 6), Point(-5, -2, 8), Point(-5, -5, 4));
	triangles[2].color.g = 100;
	triangles[2].color.b = 100;*/
	//triangles[2].mirror = true;
	//triangles[2].color.r = 240;

	sd.triangles[2] = Triangle(Point(0, 2, 5), Point(2, 0, 5), Point(-2, 0, 5));

	angle += 0.001f;
}


float findColPoint(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj) {

	float t1, nearest = INFINITY;
	bool mirror = false;

	for (int i = 0; i < sd.nSpheres; i++) {
		if (ray.intersects(sd.spheres[i], &t1, nullptr)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = sd.spheres[i].Normal(*colPoint);
				*colObj = sd.spheres + i;
				mirror = sd.spheres[i].mirror;
			}
		}
	}

	for (int i = 0; i < sd.nLights; i++) {
		if (ray.intersects(sd.lights[i], &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = sd.lights[i].Normal(*colPoint);
				*colObj = sd.lights + i;
				mirror = sd.lights[i].mirror;
			}
		}
	}

	for (int i = 0; i < sd.nTriangles; i++) {
		if (ray.intersects(sd.triangles[i], &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = sd.triangles[i].n;
				*colObj = sd.triangles + i;
				mirror = sd.triangles[i].mirror;
			}
		}
	}

	if (mirror) {
		return findColPoint(Ray(*colPoint, ray.d.Reflect(*colNormal)), colPoint, colNormal, colObj);
	}

	if (nearest < INFINITY) return true;
	return false;
}

#ifdef NONRT

float *realImg;
int iteration[THRCOUNT];

void drawPixelR(float x, float y, float *rm) {
	Point pixelPoint(x, y, 0);

	float dofStr = 0.035;
	float focalDistance = 5.0f;
	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(sd.camera, pixelPoint);

	if (dofStr > 0) {

		Triangle focalPlane = Triangle(Point(-10000, -10000, focalDistance), Point(0, 10000, focalDistance), Point(10000, -10000, focalDistance));
		ray.intersects(focalPlane, &focalDistance);

		Point focalPoint = ray.getPointFromT(focalDistance);
		float pointMove = tanf(dofStr) * focalDistance;
		Point passPoint;
		do {
			passPoint = Point(pixelPoint.x + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f) * pointMove,
				pixelPoint.y + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f) * pointMove, 0);
		} while (((Vector)(passPoint - pixelPoint)).Length() > pointMove);
		ray = Ray(passPoint, focalPoint);
	}

	float rMulR = 1.0, rMulG = 1.0, rMulB = 1.0;

	Point colPoint;

	int bounceCount = 5;

	for (bounceCount = 5; bounceCount > 0; bounceCount--) {
		if (!findColPoint(ray, &colPoint, &normal, &obj)) {
			rm[0] += 18.2 * rMulR;
			rm[1] += 42.4 * rMulG;
			rm[2] += 55.2 * rMulB;
			return;
		}
		if (obj->shape == LIGHT) {

			rm[0] += ((Light*)obj)->R() * rMulR;
			rm[1] += ((Light*)obj)->G() * rMulG;
			rm[2] += ((Light*)obj)->B() * rMulB;

			return;
		}
		ray.o = colPoint;
		rMulR *= (obj->color.r * obj->color.r) / 65100.0f;
		rMulG *= (obj->color.g * obj->color.g) / 65100.0f;
		rMulB *= (obj->color.b * obj->color.b) / 65100.0f;
		do {
			ray.d.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f;
			ray.d.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f;
			ray.d.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2) - 1.0f;
			ray.d.Normalize();
			if (ray.d * normal <= 0) ray.d = -ray.d;
		} while (ray.d * normal <= static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
	}
}

DWORD WINAPI ThreadFunc(void* data) {
	while (true) {
		//WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);

		int size = YRES / THRCOUNT;
		float rc, gc, bc;

		int limit = (int)data * size + size;
		for (int i = (int)data * size; i < limit; i++) {
			for (int j = 0; j < XRES; j++) {
				drawPixelR(j * 2.0f / YRES - XRES / (float)YRES + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / YRES
					, i * 2.0 / YRES - 1.0 + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / XRES, realImg + (i * XRES + j) * 3);

				rc = sqrtf(realImg[(i * XRES + j) * 3] / iteration[(int)data] * sd.expMultiplier);
				gc = sqrtf(realImg[(i * XRES + j) * 3 + 1] / iteration[(int)data] * sd.expMultiplier);
				bc = sqrtf(realImg[(i * XRES + j) * 3 + 2] / iteration[(int)data] * sd.expMultiplier);

				if (rc > 255) rc = 255;
				if (gc > 255) gc = 255;
				if (bc > 255) bc = 255;

				imgptr[(i * XRES + j) * 3] = rc;
				imgptr[(i * XRES + j) * 3 + 1] = gc;
				imgptr[(i * XRES + j) * 3 + 2] = bc;
			}
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
	imgptr = ptr;
	realImg = (float*)malloc(XRES * YRES * 3 * sizeof(float));
	ZeroMemory(realImg, XRES * YRES * 3 * sizeof(float));
	ZeroMemory(imgptr, XRES * YRES * 3 * sizeof(char));

	sd.nLights = 1;
	sd.nSpheres = 2;
	sd.nTriangles = 3;

	sd.spheres = (Sphere*)malloc(sd.nSpheres * sizeof(Sphere));
	sd.triangles = (Triangle*)malloc(sd.nTriangles * sizeof(Triangle));
	sd.lights = (Light*)malloc(sd.nSpheres * sizeof(Sphere));

	sd.camera = Point(0, 0, -2);
	sd.expMultiplier = 1000;

	InitFrame();

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
	for (int i = 0; i < sd.nLights; i++) {
		ray = Ray(p, sd.lights[i].c);
		if (n * ray.d > 0) {
			col = false;
			for (int j = 0; j < sd.nSpheres; j++) {
				if (sd.spheres + j != self && ray.intersects(sd.spheres[j], &t) && t > 0.001) {
					col = true;
					break;
				}
			}
			if (!col) {
				for (int j = 0; j < sd.nTriangles; j++) {
					if (sd.triangles + j != self && ray.intersects(sd.triangles[j], &t) && t > 0.001) {
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

void drawPixel(float x, float y, char *pix) {
	Point pixelPoint(x, y, 0);

	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(sd.camera, pixelPoint);

	float light;

	Point colPoint;

	if (findColPoint(ray, &colPoint, &normal, &obj)) {
		light = pointLit(colPoint, normal, obj);
		if (obj == sd.triangles + 2) {

			float v0col[] = { 1, 0, 0 };
			float v1col[] = { 0, 1, 0 };
			float v2col[] = { 0, 0, 1 };
			float retcol[] = { 0, 0, 0 };

			((Triangle*)obj)->interpolatePoint(colPoint, v0col, v1col, v2col, retcol, 3);

			pix[0] = retcol[0] * light + 8 * (1 - light);
			pix[1] = retcol[1] * light + 24 * (1 - light);
			pix[2] = retcol[2] * light + 48 * (1 - light);

		}
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

	sd.nLights = 1;
	sd.nSpheres = 2;
	sd.nTriangles = 3;

	sd.spheres = (Sphere*)malloc(sd.nSpheres * sizeof(Sphere));
	sd.triangles = (Triangle*)malloc(sd.nTriangles * sizeof(Triangle));
	sd.lights = (Light*)malloc(sd.nSpheres * sizeof(Sphere));

	sd.camera = Point(0, 0, -2);
	sd.expMultiplier = 1000;

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
