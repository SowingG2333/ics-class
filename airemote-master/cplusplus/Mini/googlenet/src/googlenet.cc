#include <string>
#include <stdio.h>

#include "googlenet.h"
#include "image_net_classes.h"

using namespace std;

int GoogleNet::Preprocess(cv::Mat& origMat, cv::Mat& out)
{
    cv::resize(origMat, out, cv::Size(width, height));
    return 0;
}

int GoogleNet::Postprocess(std::vector<AiResult>& outputs, std::any& results)
{
    auto dataBuf = (float *)outputs[0].data.get();
    if (dataBuf == nullptr) {
        return -1;
    }
    auto dataSize  = outputs[0].size;

    map<float, unsigned int, greater<float> > resultMap;
    for (uint32_t j = 0; j < dataSize / sizeof(float); ++j) {
        resultMap[*dataBuf] = j;
        dataBuf++;
    }
    
    int cnt = 0;
    int classIdx = INVALID_IMAGE_NET_CLASS_ID;
    float maxScore = 0;
    for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
        // print top 5
        if (++cnt > 5) {
            break;
        }
        printf("top %d: index[%d] value[%lf]\n", cnt, it->second, it->first);

        if (it->first > maxScore) {
            maxScore = it->first;
            classIdx = it->second;
        }
    }

    string text = "unknown";
    if (classIdx > 0 && classIdx < IMAGE_NET_CLASSES_NUM) {
        text = g_str_image_net_classes[classIdx];
    }
    cout << "Class predicted: " << text << endl;
    // return in std::any results
    results = text;

    return 0;
}
