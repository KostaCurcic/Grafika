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

float *realImg = nullptr;
int iteration[THRCOUNT];

SceneData sd;

void InitFrame()
{
	sd.genCameraCoords();
	sd.assignPointersHost();

	if (sd.reset) {
		sd.reset = false;
		for (int i = 0; i < THRCOUNT; i++) {
			iteration[i] = 0;
		}
		if(realImg != nullptr)
			memset(realImg, 0, XRES * YRES * 3 * sizeof(float));
	}
}


ColorReal traceRand(Ray ray, int iterations = 20) {
	float t1, nearest = INFINITY;
	ColorReal colorMultiplier(1, 1, 1);
	Color colGet;
	Point colPoint;
	Vector colNormal;
	GraphicsObject *colObj;
	bool mirror = false;

	if (iterations <= 0) {
		return ColorReal(0, 0, 0);
	}

	for (int i = 0; i < sd.nSpheres; i++) {
		if (ray.intersects(sd.spheres[i], &colGet, &t1, nullptr)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				colNormal = sd.spheres[i].Normal(colPoint);
				colObj = sd.spheres + i;
				mirror = sd.spheres[i].mirror;
				colorMultiplier = colGet.getRefMultiplier(sd.gamma);
			}
		}
	}

	for (int i = 0; i < sd.nLights; i++) {
		if (ray.intersects(sd.lights[i], &colGet, &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				colNormal = sd.lights[i].Normal(colPoint);
				colObj = sd.lights + i;
				colorMultiplier = colGet.getColorIntensity(sd.gamma) * sd.lights[i].intenisty;
			}
		}
	}

	for (int i = 0; i < sd.nTriangles; i++) {
		if (ray.intersects(sd.triangles[i], &colGet, &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				colNormal = sd.triangles[i].n;
				colObj = sd.triangles + i;
				mirror = sd.triangles[i].mirror;
				colorMultiplier = colGet.getRefMultiplier(sd.gamma);
			}
		}
	}

	if (nearest == INFINITY) {
		return sd.ambient.color.getColorIntensity(sd.gamma) * sd.ambient.intenisty;
	}
	else if (colObj->shape == LIGHT) {
		return colorMultiplier;
	}
	else {
		if (mirror) {
			return traceRand(Ray(colPoint, ray.d.Reflect(colNormal)), iterations - 1);
		}
		else {
			ray.o = colPoint;
			do {
				ray.d.x = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1.0f;
				ray.d.y = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1.0f;
				ray.d.z = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1.0f;
				ray.d.Normalize();
				if (ray.d * colNormal <= 0) ray.d = -ray.d;
			} while (ray.d * colNormal <= (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)));
			return colorMultiplier *= traceRand(ray, iterations - 1);
		}
	}
}

float findColPoint(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj) {

	float t1, nearest = INFINITY;
	bool mirror = false;

	for (int i = 0; i < sd.nSpheres; i++) {
		if (ray.intersects(sd.spheres[i], nullptr, &t1, nullptr)) {
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
		if (ray.intersects(sd.lights[i], nullptr , &t1)) {
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
		if (ray.intersects(sd.triangles[i], nullptr , &t1)) {
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

void drawPixelR(float x, float y, ColorReal *rm) {
	//Point pixelPoint = Point(10 + x, y, 0);

	Point pixelPoint = sd.camera + sd.c2S + sd.sR * x + sd.sD * y;

	float focalDistance = sd.focalDistance;

	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(sd.camera, pixelPoint);

	if (sd.dofStr > 0.0001f) {
		Point focalPoint = sd.camera + (Vector)(pixelPoint - sd.camera) * (1 + focalDistance / sd.camDist);

		float pointMove = tanf(sd.dofStr) * focalDistance, xOff, yOff;

		float ang = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 6.28315f;
		pointMove *= (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
		xOff = sinf(ang) * sqrtf(pointMove);
		yOff = cosf(ang) * sqrtf(pointMove);
		/*do {
			xOff = (curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 2 - 1.0f) * pointMove;
			yOff = (curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 2 - 1.0f) * pointMove;
		} while (sqrtf(xOff * xOff + yOff * yOff) > pointMove);*/
		Point passPoint = pixelPoint + sd.sR * xOff + sd.sD * yOff;
		ray = Ray(passPoint, focalPoint);
	}

	float light;
	float ra, c1, c2, c3;

	Point colPoint;

	*rm += traceRand(ray);

	return;
}

DWORD WINAPI ThreadFunc(void* data) {
	while (true) {
		//WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);

		int size = YRES / THRCOUNT;
		float rc, gc, bc;

		int limit = (int)data * size + size;

		for (int i = (int)data * size; i < limit; i++) {
			for (int j = 0; j < XRES; j++) {
				if (sd.realTime) {
					drawPixel(j * 2.0f / YRES - XRES / (float)YRES, i * 2.0 / YRES - 1.0, imgptr + (i * XRES + j) * 3);
				}
				else {
					drawPixelR(j * 2.0f / YRES - XRES / (float)YRES + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / YRES
						, i * 2.0 / YRES - 1.0 + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / XRES, (ColorReal*)realImg + (i * XRES + j));

					rc = powf(realImg[(i * XRES + j) * 3] / iteration[(int)data] * sd.expMultiplier, 1 / sd.gamma);
					gc = powf(realImg[(i * XRES + j) * 3 + 1] / iteration[(int)data] * sd.expMultiplier, 1 / sd.gamma);
					bc = powf(realImg[(i * XRES + j) * 3 + 2] / iteration[(int)data] * sd.expMultiplier, 1 / sd.gamma);

					if (rc > 255) rc = 255;
					if (gc > 255) gc = 255;
					if (bc > 255) bc = 255;

					imgptr[(i * XRES + j) * 3] = rc;
					imgptr[(i * XRES + j) * 3 + 1] = gc;
					imgptr[(i * XRES + j) * 3 + 2] = bc;
				}
			}
		}
		iteration[(int)data]++;

		//signal--;
		//WakeByAddressSingle(&signal);
	}
	return 0;
}

void InitDrawing(char * ptr)
{
	imgptr = ptr;
	realImg = (float*)malloc(XRES * YRES * 3 * sizeof(float));
	ZeroMemory(realImg, XRES * YRES * 3 * sizeof(float));
	ZeroMemory(imgptr, XRES * YRES * 3 * sizeof(char));

	InitFrame();

	for (int i = 0; i < THRCOUNT; i++) {
		iteration[i] = 1;
		HANDLE thread = CreateThread(NULL, 0, ThreadFunc, (void*)i, 0, NULL);
	}
}

float pointLit(Point &p, Vector n, GraphicsObject* self) {
	Ray ray;
	float lit = 0, t;
	bool col;
	for (int i = 0; i < sd.nLights; i++) {
		ray = Ray(p, sd.lights[i].c);
		if (n * ray.d > 0) {
			col = false;
			for (int j = 0; j < sd.nSpheres; j++) {
				if (sd.spheres + j != self && ray.intersects(sd.spheres[j], nullptr, &t) && t > 0.001) {
					col = true;
					break;
				}
			}
			if (!col) {
				for (int j = 0; j < sd.nTriangles; j++) {
					if (sd.triangles + j != self && ray.intersects(sd.triangles[j], nullptr, &t) && t > 0.001) {
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
	Point pixelPoint = sd.camera + sd.c2S + sd.sR * x + sd.sD * y;

	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(sd.camera, pixelPoint);

	float light;

	Point colPoint;

	if (findColPoint(ray, &colPoint, &normal, &obj)) {
		light = pointLit(colPoint, normal, obj);
		if (obj->shape == TRIANGLE && ((Triangle*)obj)->textured) {
			float coords[] = { 0, 0 };
			((Triangle*)obj)->interpolatePoint(colPoint, (float*)&(((Triangle*)obj)->t0), (float*)&(((Triangle*)obj)->t1), (float*)&(((Triangle*)obj)->t2), coords, 2);
			Color c = sd.textures[((Triangle*)obj)->texIndex].getColor(coords[0], coords[1]);

			pix[0] = c.r * light;
			pix[1] = c.g * light;
			pix[2] = c.b * light;
		}
		else 
		{
			pix[0] = obj->color.r * light;
			pix[1] = obj->color.g * light;
			pix[2] = obj->color.b * light;
		}
	}
	else {
		pix[0] = 40;
		pix[1] = 120;
		pix[2] = 240;

	}
}

void DrawFrame()
{
	if(sd.realTime) InitFrame();
	//signal = THRCOUNT;
	//WakeByAddressAll(&signal);

	/*while (signal > 0) {
		WaitOnAddress(&signal, &signal, sizeof(int), INFINITE);
	}*/
}

DEVICE_PREFIX void SceneData::genCameraCoords()
{
	if (camXang > 6.28318f) camXang -= 6.28318f;
	if (camXang < 0.0f) camXang += 6.28318f;
	if (camYang >= 1.5707f && camYang < 3.14159f) camYang = 1.57f;
	else {
		if (camYang < 0.0f) camYang += 6.28318f;
		//TODO doesnt work
		if (camYang > 3.141592f && camYang <= 4.712388f) camYang = 4.714f;
	}

	c2S = Vector(0, 0, 1);

	c2S = Vector(-sinf(camXang), tanf(camYang), cosf(camXang));

	c2S = c2S.Normalize() * camDist;

	sR = Vector(cosf(camXang), 0, sinf(camXang));

	sD = (c2S / camDist) % sR;

}

void SceneData::assignPointersHost() {
	for (int i = 0; i < nTriangles; i++) {
		triangles[i].tex = textures + triangles[i].texIndex;
	}
};

#endif
