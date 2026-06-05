#pragma once

#include "Triangle.h"
#include "SceneData.h"
#include <vector>
#include <string>
#include <sstream>

using namespace std;

class SceneLoader
{
public:
	SceneLoader();

	void loadObj(const char*, const Point& offset);
	void loadScene(const char* path, SceneData& sd);

	void finalize(SceneData&);

	void addLight(const Light&);
	void addSphere(const Sphere&);
	void addTriangle(const Triangle&);
	void addMaterial(const Material&);

	~SceneLoader();

private:

	void loadMaterial(const char* path, const char *name);
	bool applyMaterialProp(const string& op, stringstream& ss, Material& mat);

	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;
	std::vector<Light> lights;
	std::vector<Point> vertecies;
	std::vector<Vector> normals;
	std::vector<Material> materials;
};
