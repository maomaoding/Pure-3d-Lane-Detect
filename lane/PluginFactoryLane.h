#ifndef __PLUGIN_LAYER_FACTORY_LANE_H__
#define __PLUGIN_LAYER_FACTORY_LANE_H__

#include <cassert>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

class PluginFactoryLane : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
    bool isPlugin(const char *name);
    bool isPluginExt(const char *name);
    void destroyPlugin();
    virtual nvinfer1::IPlugin *createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) override;
    IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;
    void (*nvPluginDeleter)(INvPlugin *){[](INvPlugin *ptr) { ptr->destroy(); }};
};
#endif
