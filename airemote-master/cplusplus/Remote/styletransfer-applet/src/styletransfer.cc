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
#include "styletransfer.hpp"

using namespace std;

MAKE_APPLET_WRAPPER(StyleTransfer);

int StyleTransfer::Preprocess(cv::Mat& origMat, cv::Mat& resizeMat)
{
    origMat.convertTo(resizeMat, CV_32FC3);
    ::hwc_to_chw(resizeMat, resizeMat);
        
    return 0;
}

int StyleTransfer::Postprocess(std::vector<AiResult>& outputs, std::any& result)
{
    auto dataBuf = (float *)outputs[0].data.get();
    if (dataBuf == nullptr) {
        return -1;
    }
    auto dataSize  = outputs[0].size;

    /* chw => hwc */
    cv::Mat channels[3];

    channels[0] = cv::Mat(360, 540, CV_32FC1, dataBuf);
    channels[1] = cv::Mat(360, 540, CV_32FC1, dataBuf + 360*540);
    channels[2] = cv::Mat(360, 540, CV_32FC1, dataBuf + 360*540 + 360*540);
 
    cv::Mat mat_result;
    cv::merge(channels, 3, mat_result);
    /*  chw => hwc */

    /* return in ``std::any'' result */
    result = mat_result.clone();

    return 0;
}

int StyleTransfer::BuildAiResults(std::any& output, std::vector<AiResult>& results)
{
    auto mat = std::any_cast<cv::Mat&>(output);
    std::vector<uchar> jpeg;
    cv::imencode(".JPEG", mat, jpeg);
    const unsigned char *data_ptr = (unsigned char *) jpeg.data();
    size_t data_size = jpeg.size();

    PushResult(results, data_ptr, data_size);
    return 0;
}

