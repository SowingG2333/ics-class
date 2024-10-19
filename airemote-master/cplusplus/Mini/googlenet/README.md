An example with lib AtlasMini

1. Dependency
  sudo apt install libopencv-dev

2. Building
  cd build 
  cmake .. && make

3. Run test
  cd build 
  ln -s googlenet.om-from-somewhere .
  ln -s test.jpg-from-somewhere .
  make test

All is done.
