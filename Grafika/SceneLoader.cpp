#include "SceneLoader.h"
#include <fstream>
#include <string>
#include <sstream>

SceneLoader::SceneLoader()
{
}

void SceneLoader::loadObj(const char *path, const Point & offset)
{
	ifstream file(path);
	string line, op;

	while (getline(file, line)) {
		stringstream ss(line);
		op.clear();
		ss >> op;
		if (op == "v") {
			Point vertex;
			ss >> vertex.x >> vertex.y >> vertex.z;
			vertecies.push_back(vertex + offset);
		}
		/*else if (op == "vn") {
			Vector normal;
			ss >> normal.x >> normal.y >> normal.z;
			normals.push_back(normal);
		}*/
		else if (op == "f") {
			Triangle triangle;
			string s1, s2, s3;
			ss >> s1 >> s2 >> s3;
			triangle = Triangle(vertecies[stoi(s1.substr(0, s1.find("/"))) - 1],
								vertecies[stoi(s2.substr(0, s2.find("/"))) - 1],
								vertecies[stoi(s3.substr(0, s3.find("/"))) - 1]);
			triangles.push_back(triangle);
		}
	}
}

void SceneLoader::finalize(SceneData &sd)
{
	sd.nTriangles = triangles.size();
	sd.triangles = (Triangle*)malloc(sd.nTriangles * sizeof(Triangle));
	copy(triangles.begin(), triangles.end(), sd.triangles);

	sd.nSpheres = spheres.size();
	sd.spheres = (Sphere*)malloc(sd.nSpheres * sizeof(Sphere));
	copy(spheres.begin(), spheres.end(), sd.spheres);

	sd.nLights = lights.size();
	sd.lights = (Light*)malloc(sd.nLights * sizeof(Light));
	copy(lights.begin(), lights.end(), sd.lights);

	sd.nTextures = textures.size();
	sd.textures = (Texture*)malloc(sd.nTextures * sizeof(Texture));
	copy(textures.begin(), textures.end(), sd.textures);
}

void SceneLoader::addLight(const Light &l)
{
	lights.push_back(l);
}

void SceneLoader::addSphere(const Sphere &s)
{
	spheres.push_back(s);
}

void SceneLoader::addTriangle(const Triangle &t)
{
	triangles.push_back(t);
}

void SceneLoader::addTexture(const Texture &t)
{
	textures.push_back(t);
}

SceneLoader::~SceneLoader()
{
}
