#ifndef ATLAS_SERVICE_H
#define ATLAS_SERVICE_H

#include "airemote/airemote.h"

class AiremoteService
{
  public:
    AiremoteService (); 
    int RunServer(int port);
    int RunWorker(const char *url);
    int Stop();

    virtual int Serve(void *input, size_t size_in, void *output, size_t *size_out) = 0;
    virtual const char * HelpInfo();
};

class AtlasService : public AiremoteService 
{
  public:
    AtlasService(AtlasApplet* air) : air_(air)
    {
        ;
    }

    int Serve (void *input, size_t size_in, void *output, size_t *size_out)
    {
        return air_->Serve(input, size_in, output, size_out);    
    }

    const char * HelpInfo() {return air_->HelpInfo();}

  private:
    AtlasApplet* air_;
};


#endif
