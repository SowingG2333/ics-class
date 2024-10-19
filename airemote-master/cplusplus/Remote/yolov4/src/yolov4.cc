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
#include <string>
#include <vector>
#include <sys/stat.h>
#include <stdio.h>

#include "yolov4.hpp"
#include "post_process.hpp"

using namespace std;

int Yolov4Net::Preprocess(cv::Mat& mat, cv::Mat& resizeMat )
{
    cv::resize(mat, resizeMat, cv::Size(width, height));
    
    resizeMat.convertTo(resizeMat, CV_32FC3);
    cv::cvtColor(resizeMat, resizeMat, CV_BGR2RGB);

    resizeMat = resizeMat / 127.5 -1;

    return 0;
}

int Yolov4Net::Postprocess(std::vector<AiResult>& outputs, std::any& result)
{
    result = outputs;
    return 0;
}

int Yolov4Net::DrawBoundingBoxes(cv::Mat& input, void* outputs[], cv::Mat& result)
{
    float xScale = static_cast<float>(input.cols) / width;
    float yScale = static_cast<float>(input.rows) / height;
   
    auto outputClass = outputs[0];
    auto outputBox   = outputs[1];
    
    PostProcess postProcess(outputClass, outputBox, xScale, yScale);
    vector<BBox> boxes;
    postProcess.Process(boxes);

    postProcess.DrawBoundBoxToImage(input, boxes);

    result = input.clone();

    return 0;
}
