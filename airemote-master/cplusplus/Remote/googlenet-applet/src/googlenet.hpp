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

#ifndef GOOGLENET_H
#define GOOGLENET_H

#include "airemote/airemote.h"

class GoogleNet: public AtlasApplet {
    public:
        GoogleNet(): width(224), height(224){;}
        const char* HelpInfo(void);
	   
        int Preprocess(cv::Mat& mat, cv::Mat& intput);
        int Postprocess(std::vector<AiResult>& output, std::any& result);
	    // For AtlasApplet
        int BuildAiResults(std::any& result, std::vector<AiResult>& aiResults);
        // optional
        //void LabelClassToImage(const char*, const std::string&);

    private:
        int width, height;
};

#endif
