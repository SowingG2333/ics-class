
#include <iostream>
#include <unistd.h>
#include "styletransfer.hpp"
#include "airemote/airemote_service.h"

using namespace std;

void get_options(int argc, char** argv, char** remote, int* port, char** imgfile, int* loop);

int main (int argc, char** argv)
{
    char *remote  = NULL;
    int  port    = 0;
    char *imgfile = NULL;
    int loop      = 1;

    get_options(argc, argv, &remote, &port, &imgfile, &loop);
    if (NULL == remote || (NULL == imgfile && 0 == port)) {
        cout << ("\nUsage: main -r <file://path-to-om | tcp://url-to-om> [-p <port>] [-i <imagedir>]\n\n");
        return 1;
    }
    
    StyleTransfer air;
    air.UseRemote(remote);

    cout << air.HelpInfo() << endl;

    //for  __APPLET__
    if (port > 0) {
        air.Run(port);
        return 0;
    }
    //

    cout << "Reading image:" <<  imgfile << endl;
    cv::Mat mat_in = cv::imread(imgfile, cv::IMREAD_COLOR);
    if (mat_in.empty()) {
        cout << "Reading image failed" << endl;
        return 2;
    }
   
    std::any result;

    int rc = air.Inference(mat_in, result);        
    if (rc !=0 ) {
        cout << "[ERROR] Inferencing failed.\n";
        return 1;
    }

    auto mat_result = std::any_cast<cv::Mat&>(result);

    cv::resize(mat_result, mat_result, cv::Size(mat_in.cols, mat_in.rows));
    cv::imwrite("result.jpg", mat_result);
    
    return 0;
}

void get_options(int argc, char** argv, char** remote, int* port, char** imgfile, int* loop)
{
    int ch;

    while((ch=getopt(argc, argv, "r:p:i:l:")) != -1) {
        switch(ch) {
        case 'r':
            *remote = optarg;
            break;
        case 'p':
            *port = atoi(optarg);
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

/* Ends. */
