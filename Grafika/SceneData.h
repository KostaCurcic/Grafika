#pragma once

#include "Triangle.h"
#include "Light.h"

class SceneData
{
public:
	Point camera;
	float camXang = 0, camYang = 0, camDist = 2;

	Vector c2S, sR, sD;

	bool reset = true;
	bool realTime = true;
	bool bilinearTexture = true;
	bool useLightPredict = false;
	bool useMirrorPredict = false;

	float expMultiplier;
	unsigned short bounces = 20;

	Light ambient;

	Triangle *triangles;
	int nTriangles;

	Sphere *spheres;
	int nSpheres;

	Light *lights;
	int nLights;

	Material *materials;
	int nMaterials;

	float dofStr = 0.01f;
	float focalDistance = 5.0f;

	float gamma = 2.224f;

	DEVICE_PREFIX void genCameraCoords();
	DEVICE_PREFIX SceneData genDeviceData(Sphere *, Triangle*, Light*, Material*);
	void assignPointersHost();

};
