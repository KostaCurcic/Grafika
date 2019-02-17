#pragma once

#include "Triangle.h"
#include "SceneData.h"
#include <vector>

using namespace std;

class SceneLoader
{
public:
	SceneLoader();

	void loadObj(const char*, const Point& offset);

	void finalize(SceneData&);

	void addLight(const Light&);
	void addSphere(const Sphere&);
	void addTriangle(const Triangle&);
	void addTexture(const Texture&);

	~SceneLoader();

private:
	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;
	std::vector<Light> lights;
	std::vector<Point> vertecies;
	std::vector<Vector> normals;
	std::vector<Texture> textures;
};
