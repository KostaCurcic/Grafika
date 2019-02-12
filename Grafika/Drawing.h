#pragma once

#include "Ray.h"

#define XRES 1920
#define YRES 1080

#define NONRT

class SceneData
{
public:
	Point camera;
	float camXang = 0, camYang = 0, camDist = 2;

	Vector c2S, sR, sD;

	bool reset = true;

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

	DEVICE_PREFIX void genCameraCoords();

};

extern SceneData sd;

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();