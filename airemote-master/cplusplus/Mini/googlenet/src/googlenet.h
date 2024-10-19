#ifndef GOOGLENET_H
#define GOOGLENET_H

#include "atlasmini/atlas_mini.h"

class GoogleNet: public AtlasMini {
    public:
        GoogleNet() {width = 224; height=224;};
	   
        int Preprocess(cv::Mat& mat, cv::Mat& out);
        int Postprocess(std::vector<AiResult>& outputs, std::any& results);

    private:
        int width, height;
};

#endif
