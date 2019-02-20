#pragma once

#include "Ray.h"
#include "Texture.h"
#include "SceneData.h"

#define XRES 1280
#define YRES 720

//#define NONRT

extern SceneData sd;

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();