#include <iostream>
#include "googlenet.hpp"

using namespace std;

int main (int argc, char** argv)
{
    if (argc < 3) {
        cout << "\nUsage: " << argv[0] << " <tcp://url-to-om-service | file://path-to-om-file> <port> \n" << endl;
        return 1;
    }

    auto remote  = argv[1];
    auto port = atoi(argv[2]);
    
    GoogleNet air;
    air.UseRemote(remote);

    cout << air.HelpInfo() << endl;

    if (port > 0) {
        air.Run(port);
        return 0;
    }
    return 1;    
}

/* Ends. */
