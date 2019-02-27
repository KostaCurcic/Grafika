#include "SceneLoader.h"
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>

SceneLoader::SceneLoader()
{
}

void SceneLoader::loadMaterial(const char * path, const char *name)
{
	char pathNew[1000];
	strcpy(pathNew, path);
	char *begin = pathNew + strlen(pathNew) - 1;
	while (*begin != '\\' && *begin != '/') begin--;
	begin++;
	strcpy(begin, name);

	ifstream file(pathNew);
	string line, op;

	Material m;

	while (getline(file, line)) {
		stringstream ss(line);
		op.clear();
		ss >> op;
		if (op == "newmtl") {
			if (m.name[0] != 0) {
				materials.push_back(m);
			}
			ss >> m.name;
		}
		else if (op == "Kd") {
			ss >> m.color.r >> m.color.g >> m.color.b;
		}
	}
	if (m.name[0] != 0) {
		materials.push_back(m);
	}
}

void SceneLoader::loadObj(const char *path, const Point & offset)
{
	ifstream file(path);
	string line, op;
	vertecies.clear();

	Material *am = nullptr;

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
			if (am != nullptr) {
				triangle.mat = *am;
			}
			triangles.push_back(triangle);
		}
		else if (op == "mtllib") {
			string p;
			ss >> p;
			loadMaterial(path, p.c_str());
		}
		else if (op == "usemtl") {
			string p;
			ss >> p;
			am = nullptr;
			for (int i = 0; i < materials.size(); i++) {
				if (p == materials[i].name) {
					am = &materials[i];
					break;
				}
			}
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

	sd.nMaterials = materials.size();
	sd.materials = (Material*)malloc(sd.nMaterials * sizeof(Material));
	copy(materials.begin(), materials.end(), sd.materials);
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

void SceneLoader::addMaterial(const Material &t)
{
	materials.push_back(t);
}

SceneLoader::~SceneLoader()
{
}
