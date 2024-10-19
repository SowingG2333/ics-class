#include <iostream>
#include "googlenet.h"

using namespace std;

int main (int argc, char** argv)
{
    int rc;
    GoogleNet gnet;
     
    if (argc < 2) {
        cout << "\n Usage: gnetapp path-to-om path-to-image. \n\n" << endl;
	return 1;
    }

    cout << "Loading model ..." <<  argv[1] << endl;
    rc = gnet.UseFile(argv[1]);
    
    if (rc != 0) {
        cout << "Loading model failed. " << endl;
        return 1;
    }

    cout << "Reading image:" <<  argv[2] << endl;
    string imageFile = string(argv[2]);

    cv::Mat origMat = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (origMat.empty()) {
        cout << "Reading image failed" << endl;
        return 2;
    }
    
    cout << "Inferencing by AtlasMini:" << endl;
    std::any result;
    rc = gnet.Inference(origMat, result);
    
    if (rc !=0 ) {
        cout << "[ERROR] Inferencing failed.\n";
    }

    /* air.LabelClassToImage(result, imageFile); */
    string text = std::any_cast<string&>(result);
    cout << "Class name: " << text << endl;
    
    return 0;
}
/* Ends. */
