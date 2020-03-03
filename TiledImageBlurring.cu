#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include<time.h>

using namespace cv;
using namespace std;

#define BLUR_SIZE 3
#define BlockSize 16

// Serial implementation for running on CPU using a single thread.
void ImageBlurCpu(unsigned char* blurImg, unsigned char* InputImg,int width, int height)
{
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			
			int pixValue = 0;
			int Pixels = 0;

			for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
				for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {

					int curRow = i + blurRow;
					int curCol = j + blurCol;

					if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
						pixValue += InputImg[curRow * width + curCol];
						Pixels++;
					}
				}
			}
			blurImg[i * width + j] = (unsigned char)(pixValue / Pixels);
		}
	}
}


// The input image is grayscale and is encoded as unsigned characters [0, 255]

__global__ void ImageBlur(unsigned char *out, unsigned char *in, int width, int height) 
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if (Col < width && Row < height) {
		int pixValue = 0;
		int Pixels = 0;

		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {

				int curRow = Row + blurRow;
				int curCol = Col + blurCol;

				if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
					pixValue += in[curRow * width + curCol];
					Pixels++;
				}
			}
		}
		out[Row * width + Col] = (unsigned char)(pixValue / Pixels);
	}	
}



int main(void)
{
	cudaError_t err = cudaSuccess;

	//Read the image using OpenCV

	Mat image; //Create matrix to read iamge
	
	image= imread("Tiger.jpg",IMREAD_GRAYSCALE);
	
	if (image.empty()) {
		printf("Cannot read image file %s", "Tiger.jpg");
		exit(1);
	}

	int imageWidth=image.cols;
	int imageHeight=image.rows;

	//Allocate the host image vectors
	
	unsigned char *h_OrigImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);
	unsigned char *h_BlurImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);
	unsigned char *h_BlurImage_CPU= (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);

	h_OrigImage = image.data; //The data member of a Mat object returns the pointer to the first row, first column of the image.
							 //try image.ptr()


	//Allocate memory on the device for the original image and the blurred image and record the needed time
	
	unsigned char *d_OrigImage, *d_BlurImage = NULL;
	float imageSize = imageHeight * imageWidth * sizeof(unsigned char);

	GpuTimer timer;
	timer.Start();
	
	//@@ Insert Your code Here to allocate memory on the device for original and blurred images
	
	err = cudaMalloc((void **)&d_OrigImage, imageSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Original Image (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_BlurImage, imageSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Blur Image (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	
	
	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());

	

	//Copy the original image from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();
	
	//@@ Insert your code here to Copy the original image from the host to the device

	cudaMemcpy(d_OrigImage, h_OrigImage, imageSize, cudaMemcpyHostToDevice);

	timer1.Stop();
	printf("Time to copy the Original image from the host to the device is: %f msecs.\n", timer1.Elapsed());

	
	//Do the Processing on the GPU
	//Kernel Execution Configuration Parameters
	dim3 dimBlock(16, 16, 1);
	
	//@@ Insert Your code Here for grid dimensions
	
	dim3 gridDim((imageWidth - 1) / BlockSize + 1, (imageHeight - 1) / BlockSize + 1, 1);
	
	//Invoke the ImageBlur kernel and record the needed time for its execution
	//GpuTimer timer;
	GpuTimer timer2;
	timer2.Start();

	//@@ Insert your code here for kernel invocation

	ImageBlur << < gridDim, dimBlock >> > (d_BlurImage, d_OrigImage, imageWidth, imageHeight);

	timer2.Stop();
	printf("Implemented ImageBlur Kernel ran in: %f msecs.\n", timer2.Elapsed());

	//Copy resulting blurred image from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();
	
	//@@ Insert your code here to Copy resulting blurred image from device to host 

	cudaMemcpy(h_BlurImage, d_BlurImage, imageSize, cudaMemcpyDeviceToHost);

	timer3.Stop();
	printf("Time to copy the blurred image from the device to the host is: %f msecs.\n", timer3.Elapsed());

	

	//Do the Processing on the CPU
	clock_t begin = clock();
	
	//@@ Insert your code her to call the cpu function for ImageBlur on the CPU	

	ImageBlurCpu(h_BlurImage_CPU, h_OrigImage, imageWidth, imageHeight);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC*1000;
	printf("Implemented CPU code ran in: %f msecs.\n", time_spent);

	//Postprocess and Display the resulting images using OpenCV
	Mat Image1(imageHeight, imageWidth,CV_8UC1,h_BlurImage); //grayscale image mat object
	Mat Image2(imageHeight,imageWidth,CV_8UC1,h_BlurImage_CPU ); //grayscale image mat object

	

	namedWindow("CPUImage", WINDOW_NORMAL); //Create window to display the image
	namedWindow("GPUImage", WINDOW_NORMAL);
	namedWindow("OriginalImage", WINDOW_NORMAL);
	imshow("GPUImage",Image1);
	imshow("CPUImage",Image2); //Display the image in the window
	imshow("OriginalImage", image); //Display the original image in the window
	waitKey(0); //Wait till you press a key 

	
	
	//Free host memory
	image.release();
	Image1.release();
	Image2.release();
	free(h_BlurImage);
	free(h_BlurImage_CPU);

	//Free device memory
	
	//@@ Insert your code here to free device memory

	cudaFree(d_OrigImage);
	cudaFree(d_BlurImage);

	return 0;

}