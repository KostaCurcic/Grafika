#pragma once

#include "Ray.h"
#include "Texture.h"
#include "SceneData.h"

#define XRES 1920
#define YRES 1080

//#define NONRT

extern SceneData sd;

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();