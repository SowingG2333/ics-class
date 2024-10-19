/**
 Inferemote: a Remote Inference Toolkit for Ascend 310

 Copyright (c) 2022 Jiasheng Hao <haojiash@qq.com>
 (University of Electronic Science and Technology of China, UESTC)

 Permission  is  hereby  granted,  free  of  charge,  to  any  person
 obtaining a copy of this software and associated documentation files
 (the  "Software"),  to deal  in  the  Software without  restriction,
 including without limitation the rights to use, copy, modify, merge,
 publish, distribute, sublicense, and/or sell copies of the Software,
 and to  permit persons to whom  the Software is furnished  to do so,
 subject to the following conditions:

 The  above copyright  notice  and this  permission  notice shall  be
 included in all copies or substantial portions of the Software.

 THE  SOFTWARE IS  PROVIDED "AS  IS", WITHOUT  WARRANTY OF  ANY KIND,
 EXPRESS OR IMPLIED,  INCLUDING BUT NOT LIMITED TO  THE WARRANTIES OF
 MERCHANTABILITY,    FITNESS   FOR    A   PARTICULAR    PURPOSE   AND
 NONINFRINGEMENT. IN NO EVENT SHALL  THE AUTHORS OR COPYRIGHT HOLDERS
 BE LIABLE FOR  ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
 ACTION OF  CONTRACT, TORT OR OTHERWISE,  ARISING FROM, OUT OF  OR IN
 CONNECTION WITH  THE SOFTWARE OR  THE USE  OR OTHER DEALINGS  IN THE
 SOFTWARE.
**/

#include <dirent.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string>

#include "googlenet.hpp"
#include "image_net_classes.h"

using namespace std;
namespace {
    const uint32_t kTopNConfidenceLevels = 5;
    const uint32_t kOutputDataBufId = 0;
}

int GoogleNet::Preprocess(cv::Mat& image, cv::Mat& input)
{
    cv::resize(image, input, cv::Size(width, height));
    return 0;
}

int GoogleNet::Postprocess(std::vector<AiResult>& outputs, std::any& results)
{
    auto dataBuf = (float *)outputs[kOutputDataBufId].data.get();
    if (dataBuf == nullptr) {
        return -1;
    }
    auto dataSize  = outputs[kOutputDataBufId].size;

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
        if (++cnt > kTopNConfidenceLevels) {
            break;
        }
        printf("top %d: index[%d] value[%lf]\n", cnt, it->second, it->first);

        if (it->first > maxScore) {
            maxScore = it->first;
            classIdx = it->second;
        }
    }

    string text = "unknown";
    if (classIdx > 0 && classIdx <= IMAGE_NET_CLASSES_NUM) {
        text = g_str_image_net_classes[classIdx];
    }
    cout << "Class predicted: " << text << endl; 
    // return in std::any results
    results = text;

    return 0;
}

void GoogleNet::LabelClassToImage(const char *className, const string& origImagePath)
{
    // generate outfile name
    int pos = origImagePath.find_last_of("/");
    string filename(origImagePath.substr(pos + 1));

    stringstream sstream;
    sstream.str("");
    sstream << "./out_"  << filename;
    string outputPath = sstream.str();

    // write outfile 
    int fontFace = 3;
    double fontScale = 1;
    int thickness = 2;
    cv::Point origin;
    origin.x = 10;
    origin.y = 50;

    auto text = string(className);
    cv::Mat resultImage = cv::imread(origImagePath, cv::IMREAD_COLOR);
    cv::putText(resultImage, text, origin, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness, 4, 0);
    cv::imwrite(outputPath, resultImage);
}

