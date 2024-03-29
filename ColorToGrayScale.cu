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


//In OpenCV the image is read in BGR format, that is for each pixel, the Blue, Green, then Red components are read from the image file.

// Serial implementation for running on CPU using a single thread.
void rgbaToGrayscaleCpu(unsigned char* rgbImage, unsigned char* grayImage, int numRows, int numCols, int Channels)
{

	//@@ Insert your code here
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			int grayOffset = numCols*i + j;
			int rgbOffset = grayOffset * 3;
			unsigned char b = rgbImage[rgbOffset];
			unsigned char g = rgbImage[rgbOffset + 1];
			unsigned char r = rgbImage[rgbOffset + 2];
			grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
		}
	}
}


// we have 3 channels corresponding to B, G, and R components of each pixel
// The input image is encoded as unsigned characters [0, 255]

__global__ void colorToGrayscaleConversion(unsigned char * Pout, unsigned
	char * Pin, int width, int height, int numChannels)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		int grayOffset = y*width + x;
		int rgbOffset = grayOffset * 3;

		unsigned char b = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char r = Pin[rgbOffset + 2];

		Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
	}
}



int main(void)
{
	cudaError_t err = cudaSuccess;
	//Read the image using OpenCV
	Mat image; //Create matrix to read iamge
	image = imread("Scene.jpeg", IMREAD_COLOR);
	if (image.empty()) {
		printf("Cannot read image file %s", "Scene.jpeg");
		exit(1);
	}


	int imageChannels = 3;
	int imageWidth = image.cols;
	int imageHeight = image.rows;

	//Allocate the host image vectors
	unsigned char *h_rgbImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	unsigned char *h_grayImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);
	unsigned char *h_grayImage_CPU = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);

	h_rgbImage = image.data; //The data member of a Mat object returns the pointer to the first row, first column of the image.
							 //try image.ptr()


	//Allocate memory on the device for the rgb image and the grayscale image and record the needed time
	unsigned char *d_rgbImage = NULL;
	unsigned char *d_grayImage = NULL;
	float imageSize = imageHeight * imageWidth * imageChannels;
	GpuTimer timer;
	timer.Start();

	//@@ Insert Your code Here to allocate memory on the device for color and gray images

	err = cudaMalloc((void **)&d_rgbImage, imageSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device rgbImage (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_grayImage, imageSize / 3);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device grayImage (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());



	//Copy the rgb image from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();

	//@@ Insert your code here to Copy the rgb image from the host to the device

	cudaMemcpy(d_rgbImage, h_rgbImage, imageSize, cudaMemcpyHostToDevice);

	timer1.Stop();
	printf("Time to copy the RGB image from the host to the device is: %f msecs.\n", timer1.Elapsed());


	//Do the Processing on the GPU
	//Kernel Execution Configuration Parameters
	dim3 dimBlock(16, 16, 1);

	//@@ Insert Your code Here for grid dimensions
	dim3 gridDim((imageWidth - 1) / 16 + 1, (imageHeight - 1) / 16 + 1, 1);

	//Invoke the colorToGrayscaleConversion kernel and record the needed time for its execution
	//GpuTimer timer;
	GpuTimer timer2;
	timer2.Start();

	//@@ Insert your code here for kernel invocation

	colorToGrayscaleConversion << < gridDim, dimBlock >> > (d_grayImage, d_rgbImage, imageWidth, imageHeight, imageChannels);

	timer2.Stop();
	printf("Implemented CUDA code ran in: %f msecs.\n", timer2.Elapsed());

	//Copy resulting gray image from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();

	//@@ Insert your code here to Copy resulting gray image from device to host 
	float imageSize_gray = imageHeight * imageWidth;
	cudaMemcpy(h_grayImage, d_grayImage, imageSize_gray, cudaMemcpyDeviceToHost);

	timer3.Stop();
	printf("Time to copy the Gray image from the device to the host is: %f msecs.\n", timer3.Elapsed());



	//Do the Processing on the CPU
	clock_t begin = clock();

	//@@ Insert your code her to call the cpu function for colortograyscale conversion on the CPU	
	
	rgbaToGrayscaleCpu(h_rgbImage, h_grayImage_CPU, imageWidth, imageHeight, imageChannels);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Implemented CPU code ran in: %f msecs.\n", time_spent);

	//Postprocess and Display the resulting images using OpenCV
	Mat Image1(imageHeight, imageWidth, CV_8UC1, h_grayImage); //grayscale image mat object
	Mat Image2(imageHeight, imageWidth, CV_8UC1, h_grayImage_CPU); //grayscale image mat object



	namedWindow("CPUImage", WINDOW_NORMAL); //Create window to display the image
	namedWindow("GPUImage", WINDOW_NORMAL);
	imshow("GPUImage", Image1);
	imshow("CPUImage", Image2); //Display the image in the window
	waitKey(0); //Wait till you press a key 



				//Free host memory
	image.release();
	Image1.release();
	Image2.release();
	free(h_grayImage);
	free(h_grayImage_CPU);

	//Free device memory

	//@@ Insert your code here to free device memory

	free(d_grayImage);
	free(d_rgbImage);

	return 0;

}