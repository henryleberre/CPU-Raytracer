@echo Off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x86
@echo On
cl rtx.cpp /std:c++17
rtx.exe
frame.ppm