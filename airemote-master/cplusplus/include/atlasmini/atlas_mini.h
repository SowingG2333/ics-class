#ifndef ATLASMINI_H
#define ATLASMINI_H

#include <any>
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "airemote/airtypes.h"

using namespace std;  

class AtlasMini : public InferenceModel {

  public:
    AtlasMini(int deviceId = 0);
    ~AtlasMini();

    int UseFile(const char* modelPath = NULL);

    int Inference (cv::Mat& input, std::any& output);
    const char* GetModelString ();

  private:
    void *handle;
    char modelPath[256];
};

#endif
