# Batsher

Folder structure: `sherpa.bat/cpp_interface/Batsher`

Add new folder: `sherpa.bat/cpp_interface/output`

Modify line 7 in `sherpa.bat/cpp_interface/CMakeLists.txt` to the following:  
`add_executable(test Batsher/test.cpp Rambo.cpp Rambo.hpp root_finder.hpp)`

To generate events:
1. go to `bat.sherpa/cpp_interface`
2. run `cd build; make; cd ..`
3. run ` ./build/test` or ` ./build/test julia OUTPUT=0`
