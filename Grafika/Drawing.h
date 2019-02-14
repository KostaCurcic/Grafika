#pragma once

#include "Ray.h"
#include "Texture.h"

#define XRES 1920
#define YRES 1080

//#define NONRT

class SceneData
{
public:
	Point camera;
	float camXang = 0, camYang = 0, camDist = 2;

	Vector c2S, sR, sD;

	bool reset = true;
	bool realTime = false;
	bool bilinearTexture = true;

	float expMultiplier;

	Light ambient;

	Triangle *triangles;
	int nTriangles;

	Sphere *spheres;
	int nSpheres;

	Light *lights;
	int nLights;

	Texture *textures;
	int nTextures;

	float dofStr = 0.01f;
	float focalDistance = 5.0f;

	float gamma = 2.224f;

	DEVICE_PREFIX void genCameraCoords();
	DEVICE_PREFIX SceneData genDeviceData(Sphere *, Triangle*, Light*, Texture*);
	void assignPointersHost();

};

extern SceneData sd;

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();