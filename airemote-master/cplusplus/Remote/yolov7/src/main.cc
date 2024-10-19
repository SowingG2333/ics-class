#include <iostream>
#include "yolov7.h"

int main (int argc, char** argv)
{
    if (argc < 3) {
        cout << "\nUsage: " << argv[0] << " <tcp://url-to-om-service | file://path-to-om-file> <path-to-picture-file> \n" << endl;
        return 1;
    }

    auto remote  = argv[1];
    auto imgfile = argv[2];
    
    cout << "Reading image:" <<  imgfile << endl;
    cv::Mat image = cv::imread(imgfile, cv::IMREAD_COLOR);
    if (image.empty()) {
        cout << "Reading image failed" << endl;
        return 2;
    }

    YoloV7 air;
    air.UseRemote(remote);

    std::any result;

    int rc = air.Inference(image, result);
    if (rc !=0 ) {
        cout << "[ERROR] Inferencing failed.\n";
        return 1;
    }

    auto pointers = std::any_cast<std::vector<AiResult>&>(result);
    auto classBuf = (float *)pointers[0].data.get();

    air.DrawBBoxes(image, classBuf);

    cv::imwrite("result.jpg", image);
    
    return 0;
}

/* Ends. */
