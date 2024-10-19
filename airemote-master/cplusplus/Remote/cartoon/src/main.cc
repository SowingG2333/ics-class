#include <unistd.h>
#include <iostream>
#include "cartoon.h"

using namespace std;

void get_options(int argc, char** argv, char **remote, char **imgfile, int *loop);


int main (int argc, char** argv)
{
    char *remote  = NULL;
    char *imgfile = NULL;
    int loop      = 1;

    get_options(argc, argv, &remote, &imgfile, &loop);       
    if (NULL == remote || NULL == imgfile) {
        cout << ("\nUsage: main -r <file://path-to-om | tcp://url-to-om> -i <imagedir>\n");
    	return 1;
    }

    string imageFile = string(imgfile);
    cout << "Reading image:" <<  imageFile << endl;

    cv::Mat origMat = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (origMat.empty()) {
        cout << "Reading image failed" << endl;
        return 2;
    }
    
    Cartoon air;
    air.UseRemote(remote);

    std::any result;

    int rc = air.Inference(origMat, result); 
    if (rc !=0 ) {
        cout << "[ERROR] Inferencing failed.\n";
    }	
    
    auto mat_result = std::any_cast<cv::Mat&>(result);
    cv::resize(mat_result, mat_result, cv::Size(origMat.cols, origMat.rows));
    cv::imwrite("result.jpg", mat_result);

    return 0;
}

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
/* Ends. */
