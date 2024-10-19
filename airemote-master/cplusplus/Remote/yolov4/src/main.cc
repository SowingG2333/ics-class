#include <unistd.h>
#include <iostream>
#include "yolov4.hpp"

using namespace std;

void get_options(int argc, char** argv, char **remote, char **imgfile, int *loop)
{
    int ch;

    while((ch=getopt(argc, argv, "r:i:l:")) != -1) {
        switch(ch) {
            case 'r':
                *remote = optarg;
                break;
            case 'i':
                *imgfile = optarg;
                break;
            case 'l':
                *loop = atoi(optarg);
                break;
        }
    }
}

int main (int argc, char** argv)
{
    char *remote  = NULL;
    char *imgfile = NULL;
    int loop      = 1;

    get_options(argc, argv, &remote, &imgfile, &loop);       
    if (NULL == remote || NULL == imgfile) {
        cout << ("\nUsage: main -r <file://path-to-om | tcp://url-to-om> -i <imagedir> \n\n");
    	return 1;
    }

    cout << "Reading image:" <<  imgfile << endl;

    cv::Mat origMat = cv::imread(imgfile, cv::IMREAD_COLOR);
    if (origMat.empty()) {
        cout << "Reading image failed" << endl;
        return 2;
    }
    
    Yolov4Net air;
    air.UseRemote(remote);
    
    /* two pointers about data of the BBoxes */
    void* outputs[2];
    //std::any result = outputs;
    std::any result;

    int rc = air.Inference(origMat, result);
    if (rc !=0 ) {
        cout << "[ERROR] Inferencing failed.\n";
    }

    auto pointers = std::any_cast<std::vector<AiResult>&>(result);

    outputs[0] = (void *)pointers[0].data.get();
    outputs[1] = (void *)pointers[1].data.get();

    if (outputs[0] == nullptr || outputs[1] == nullptr) {
        return -1;
    }
    
    cv::Mat mat_result;
    rc = air.DrawBoundingBoxes(origMat, outputs, mat_result);
    cv::imwrite("result.jpg", mat_result);
    
    return 0;
}
/* Ends. */
