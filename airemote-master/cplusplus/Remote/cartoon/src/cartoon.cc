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

#include "cartoon.h"

using namespace std;

int Cartoon::Preprocess(cv::Mat& mat, cv::Mat& resizeMat)
{
    cv::resize(mat, resizeMat, cv::Size(width, height)); // model input size
    resizeMat.convertTo(resizeMat, CV_32FC3);
    resizeMat = resizeMat / 127.5 -1;
    cv::cvtColor(resizeMat, resizeMat, CV_BGR2RGB);
    
    return 0;
}

int Cartoon::Postprocess(std::vector<AiResult>& outputs, std::any& result)
{
    void *data;
    uint32_t dataSize;

    data = (void *)outputs[0].data.get();
    if (data == nullptr) {
        return -1;
    }
    dataSize  = outputs[0].size;

    unsigned char* outData = NULL;
    outData = reinterpret_cast<unsigned char *>(data);
    uint32_t size = static_cast<uint32_t>(dataSize) / sizeof(float);

    cv::Mat mat_result(256, 256, CV_32FC3, const_cast<float*>((float*)outData));
    cv::cvtColor(mat_result, mat_result, CV_RGB2BGR);
    mat_result = (mat_result + 1) * 127.5;
    
    result = mat_result.clone();

    return 0;
}
