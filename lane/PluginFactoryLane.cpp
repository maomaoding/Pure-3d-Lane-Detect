#include "PluginFactoryLane.h"
#include <vector>
#include <algorithm>
using namespace nvinfer1;
using namespace nvcaffeparser1;

/******************************/
// PluginFactory
/******************************/
nvinfer1::IPlugin *PluginFactoryLane::createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights)
{
    assert(isPlugin(layerName));
    return nullptr;
}

IPlugin *PluginFactoryLane::createPlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
    return nullptr;
}

bool PluginFactoryLane::isPlugin(const char *name)
{
    return isPluginExt(name);
}
bool PluginFactoryLane::isPluginExt(const char *name)
{
    return 0;
}

void PluginFactoryLane::destroyPlugin()
{
}
