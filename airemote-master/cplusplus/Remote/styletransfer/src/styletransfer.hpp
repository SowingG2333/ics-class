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

#ifndef STYLE_H
#define STYLE_H

#include <airemote/airemote.h>

// -------------------------------------------------------------------
// Comment the following line to build standalone application
#define __APPLET__
// -------------------------------------------------------------------

// App or Applet?
#ifdef __APPLET__
class StyleTransfer: public AtlasApplet {
#else
class StyleTransfer: public AtlasRemote {
#endif
    public:
        StyleTransfer(): width(1080), height(720) {;}
        const char* HelpInfo () {return "StyleTransfer Applet";} 

        
        int Preprocess(cv::Mat& image, cv::Mat& intput);
        int Postprocess(std::vector<AiResult>& outputs, std::any& results);
#ifdef __APPLET__
        int BuildAiResults(std::any& output, std::vector<AiResult>& results);
#endif
    private:
        int width, height;
};

#endif
