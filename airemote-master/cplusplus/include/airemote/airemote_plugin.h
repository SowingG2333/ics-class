#ifndef AIR_PLUGIN_H
#define AIR_PLUGIN_H
#ifdef __ATLAS_PLUGIN__

#include "airemote/airemote.h"
static AtlasApplet *air = NULL;

AtlasApplet* CreateAtlasApplet()
{
    return new __ATLAS_PLUGIN__();
};

extern "C" {

void* Load()
{
    if (air) return air;

    air = CreateAtlasApplet();

    return air;
}

int use_remote(const char *remote)
{
    if (!air) return -1;
 
    int rc;
    rc = air->UseRemote(remote);

    return rc;
}

int inference_remote(void *input, size_t size, void *result, size_t *len)
{
    if (!air || !result) return -1;

    int rc = air->Serve(input, size, result, len);

    return 0;
}

void Unload()
{
    if (air) {
       delete air;
    }
}

} //extern "C"

#endif // #ifdef
#endif // #ifdef
