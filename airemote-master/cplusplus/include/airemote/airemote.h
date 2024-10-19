#ifndef AIREMOTE_H
#define AIREMOTE_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "airemote/airtypes.h"

using namespace std;

class AtlasRemote : public InferenceModel {

  public:
    AtlasRemote();
    virtual ~AtlasRemote();

    int UseRemote(const char *remote);         
    int Inference(cv::Mat& input, std::any& result);
    const char* GetModelString ();
    virtual const char* HelpInfo();

  protected:
    int run_remote(const unsigned char* input, size_t size, 
                   std::vector<AiResult>& results);
    //virtual int PreprocessCaller (cv::Mat& intput, cv::Mat& output);
    //virtual int PostprocessCaller(std::vector<AiResult>& results, void* output, size_t* size);
    virtual int ServeInfo(char* output, size_t* size_out);
    int GetRemoteInfo(char* output, size_t* size_out);
    
  private:
    void *handle_;
};

class AtlasApplet : public AtlasRemote {
  public:
    virtual int BuildAiResults(std::any& result, std::vector<AiResult>& aiResults) = 0;
    void PushResult(std::vector<AiResult>& results, const void* data, size_t len);
     // for AiremoteService
    int Serve(void* input, size_t size_in, void* output, size_t* size_out);
    virtual int ServeInfo(char* output, size_t* size_out);
    int Run(int port);

  protected:
    int PostprocessCaller(std::vector<AiResult>& results, void* output, size_t* size);
    int ServeImage(void* input, size_t size_in, void* output, size_t* size_out);
    int PackInfo(char *info, void* output, size_t* size_out);
};

void hwc_to_chw(cv::InputArray src, cv::OutputArray dst);
void chw_to_hwc(float* buffer, int rows, int cols, cv::OutputArray dst);

#endif
