## CUDA Programming 

### Content 

- [GPUtimer](https://github.com/BahaaKaaki/CUDA/blob/master/GpuTimer.h) : the timer used in the following codes.

- [VectorMultiply1](https://github.com/BahaaKaaki/CUDA/blob/master/VectorMultiply1.cu) : This code performs element-wise vector multiplication.
- [VectorMultiply2](https://github.com/BahaaKaaki/CUDA/blob/master/VectorMultiply2.cu) : This code performs element-wise vector multiplication where each thread calculated two elements of the output vector. 
- [ColorToGrayscale](https://github.com/BahaaKaaki/CUDA/blob/master/ColorToGrayScale.cu) :  This code uses OpenCV libraries, which is an Open Source Computer Vision Library, to read the input color image and then display the resulting Grayscale image. 
You need to include both property sheets for both debug and release : [DebugPropertySheet.props](https://github.com/BahaaKaaki/CUDA/blob/master/DebugPropertySheet.props) & [PropertySheetRelease.props](https://github.com/BahaaKaaki/CUDA/blob/master/PropertySheetRelease.props) then adjust their corresponding paths.
The images used are : [lena_color.bmp](https://github.com/BahaaKaaki/CUDA/blob/master/lena_color.bmp), [Island.jpeg](https://github.com/BahaaKaaki/CUDA/blob/master/Island.jpeg) and [Scene.jpeg](https://github.com/BahaaKaaki/CUDA/blob/master/Scene.jpeg).
- [BasicMatrixMultiplication](https://github.com/BahaaKaaki/CUDA/blob/master/BasicMatrixMultiplication.cu) : This code implement basic dense matrix multiplication routine.
- [TiledMatrixMultiplication](https://github.com/BahaaKaaki/CUDA/blob/master/TiledMatrixMUltiplication.cu) : This code implement tiled dense matrix multiplication routine using shared memory.

### References
OpenCV Tutorials: 
https://docs.opencv.org/3.0beta/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html
