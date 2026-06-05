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

// Handles the material-related keys shared by every object block. Returns true
// if it recognised (and consumed) the keyword, so the block loops can fall
// through to it for anything they don't handle themselves.
bool SceneLoader::applyMaterialProp(const string &op, stringstream &ss, Material &mat)
{
	if (op == "color") {
		ss >> mat.color.r >> mat.color.g >> mat.color.b;
	}
	else if (op == "mirror") {
		mat.mirror = true;
	}
	else if (op == "transparent") {
		mat.transparent = true;
	}
	else if (op == "refindex") {
		ss >> mat.refIndex;
	}
	else if (op == "texture") {
		string p;
		ss >> p;
		mat.texture.load(p.c_str());
		mat.color = ColorReal(1, 1, 1);
	}
	else if (op == "material") {
		string name;
		ss >> name;
		for (int i = 0; i < materials.size(); i++) {
			if (name == materials[i].name) {
				mat = materials[i];
				break;
			}
		}
	}
	else {
		return false;
	}
	return true;
}

void SceneLoader::loadScene(const char *path, SceneData &sd)
{
	ifstream file(path);
	string line, op;

	while (getline(file, line)) {
		stringstream ss(line);
		op.clear();
		ss >> op;

		if (op.empty() || op[0] == '#') continue;

		else if (op == "camera") {
			ss >> sd.camera.x >> sd.camera.y >> sd.camera.z;
		}
		else if (op == "exposure") {
			ss >> sd.expMultiplier;
		}
		else if (op == "dof") {
			ss >> sd.dofStr;
		}
		else if (op == "focal") {
			ss >> sd.focalDistance;
		}
		else if (op == "gamma") {
			ss >> sd.gamma;
		}
		else if (op == "ambient") {
			ss >> sd.ambient.mat.color.r >> sd.ambient.mat.color.g >> sd.ambient.mat.color.b >> sd.ambient.intenisty;
		}
		else if (op == "obj") {
			string p;
			Point offset;
			ss >> p >> offset.x >> offset.y >> offset.z;
			loadObj(p.c_str(), offset);
		}
		else if (op == "light") {
			Sphere s;
			float intensity = 1;
			while (getline(file, line)) {
				stringstream bs(line);
				string bop;
				bs >> bop;
				if (bop.empty() || bop[0] == '#') continue;
				if (bop == "end") break;
				if (bop == "center") bs >> s.c.x >> s.c.y >> s.c.z;
				else if (bop == "radius") bs >> s.r;
				else if (bop == "intensity") bs >> intensity;
				else applyMaterialProp(bop, bs, s.mat);
			}
			lights.push_back(Light(s, intensity));
		}
		else if (op == "sphere") {
			Sphere s;
			while (getline(file, line)) {
				stringstream bs(line);
				string bop;
				bs >> bop;
				if (bop.empty() || bop[0] == '#') continue;
				if (bop == "end") break;
				if (bop == "center") bs >> s.c.x >> s.c.y >> s.c.z;
				else if (bop == "radius") bs >> s.r;
				else if (bop == "cut") {
					s.cut = true;
					bs >> s.cutPoint.x >> s.cutPoint.y >> s.cutPoint.z;
					bs >> s.cutVector.x >> s.cutVector.y >> s.cutVector.z;
				}
				else applyMaterialProp(bop, bs, s.mat);
			}
			spheres.push_back(s);
		}
		else if (op == "triangle") {
			Point v0, v1, v2, uv0, uv1, uv2;
			Material mat;
			while (getline(file, line)) {
				stringstream bs(line);
				string bop;
				bs >> bop;
				if (bop.empty() || bop[0] == '#') continue;
				if (bop == "end") break;
				if (bop == "v0") bs >> v0.x >> v0.y >> v0.z;
				else if (bop == "v1") bs >> v1.x >> v1.y >> v1.z;
				else if (bop == "v2") bs >> v2.x >> v2.y >> v2.z;
				else if (bop == "uv0") bs >> uv0.x >> uv0.y >> uv0.z;
				else if (bop == "uv1") bs >> uv1.x >> uv1.y >> uv1.z;
				else if (bop == "uv2") bs >> uv2.x >> uv2.y >> uv2.z;
				else applyMaterialProp(bop, bs, mat);
			}
			Triangle t = Triangle(v0, v1, v2);
			t.mat = mat;
			t.t0 = uv0;
			t.t1 = uv1;
			t.t2 = uv2;
			triangles.push_back(t);
		}
		else if (op == "material") {
			Material m;
			ss >> m.name;
			while (getline(file, line)) {
				stringstream bs(line);
				string bop;
				bs >> bop;
				if (bop.empty() || bop[0] == '#') continue;
				if (bop == "end") break;
				applyMaterialProp(bop, bs, m);
			}
			materials.push_back(m);
		}
	}

	sd.genCameraCoords();
	finalize(sd);
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
