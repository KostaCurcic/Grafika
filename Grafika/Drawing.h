#pragma once

#include "Ray.h"

#define XRES 1920
#define YRES 1080

//#define NONRT

struct SceneData
{
	Point camera;
	float expMultiplier;

	Triangle *triangles;
	int nTriangles;

	Sphere *spheres;
	int nSpheres;

	Light *lights;
	int nLights;

	float dofStr = 0.01f;
	float focalDistance = 5.0f;

	float gamma = 2.224f;

} typedef SceneData;

extern SceneData sd;

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();