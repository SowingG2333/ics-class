#include <iostream>
#include "googlenet.hpp"

int main (int argc, char** argv)
{
    auto remote  = argv[1];
    auto imgfile = argv[2];
    
    GoogleNet air;
    air.UseRemote(remote);
    
    cout << "Reading image:" <<  imgfile << endl;
    cv::Mat origMat = cv::imread(imgfile, cv::IMREAD_COLOR);
    if (origMat.empty()) {
        cout << "Reading image failed" << endl;
        return 2;
    }

    std::any result;
    
    auto rc = air.Inference(origMat, result);
    if (rc !=0 ) {
        cout << "[ERROR] Inferencing failed.\n";
        return 1;
    }
    auto text = std::any_cast<string&>(result);
    cout << "Class name: " << text << endl;
    air.LabelClassToImage(text.c_str(), imgfile); 
    
    return 0;
}

/* Ends. */
