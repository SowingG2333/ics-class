#ifndef AIRTYPES_H
#define AIRTYPES_H

#include <any>
#include <vector>
#include <memory>
#include <string>

using namespace std;

struct AiResult {
    std::shared_ptr<void> data = nullptr;
    uint32_t size;
};

#define SHARED_PRT_U8_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { delete[](p); }))

class InferenceModel {
    public:
    virtual int Inference (cv::Mat& input, std::any& result) = 0;

    virtual int Preprocess (cv::Mat& image, cv::Mat& input) = 0;
    virtual int Postprocess(std::vector<AiResult>& outputs, std::any& results) = 0;

    virtual const char* GetModelString () = 0;
};

#endif
