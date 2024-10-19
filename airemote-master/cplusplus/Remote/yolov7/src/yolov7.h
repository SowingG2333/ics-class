#include "airemote/airemote.h"

class YoloV7: public AtlasRemote {
    public:
        YoloV7(): width(640), height(640){;}
        //const char* HelpInfo(void);

        int Preprocess(cv::Mat& mat, cv::Mat& intput);
        int Postprocess(std::vector<AiResult>& output, std::any& result);
        // For AtlasApplet
        //int BuildAiResults(std::any& result, std::vector<AiResult>& aiResults);
        // optional
        //void LabelClassToImage(const char*, const std::string&);
        int DrawBBoxes(cv::Mat& srcImage, float* classBuff);

    private:
        int width, height;
};

