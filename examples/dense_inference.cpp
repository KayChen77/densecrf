//reviewed and modified by cgq, 07/28/2016

/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "densecrf.h"
#include <cstdio>
#include <cmath>
#include "util.h"

#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void preprocessing(const char* backgroundRGB, const char* backgroundDepth,
	const char* imageRGB, const char* imageDepth, const char* outputImage, const char* outputAnno)
{
	int width = 1920;
	int height = 1080;

	uchar *output = new uchar[width * height];

	cv::Mat image_background;
	image_background = cv::imread(backgroundRGB, CV_LOAD_IMAGE_COLOR);
	//cv::cvtColor(image_background, image_background, CV_BGR2GRAY);

	cv::Mat image;
	image = cv::imread(imageRGB, CV_LOAD_IMAGE_COLOR);
	////
	cv::Mat depth_background;
	depth_background = cv::imread(backgroundDepth, CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat depth;
	depth = cv::imread(imageDepth, CV_LOAD_IMAGE_UNCHANGED);

	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
	compression_params.push_back(0);

	cv::imwrite(outputImage, image, compression_params);

	for (int v = 0; v < height; v++){
		for (int u = 0; u < width; u++){

			int Ir_background = image_background.at<cv::Vec3b>(v, u)[2];
			int Ig_background = image_background.at<cv::Vec3b>(v, u)[1];
			int Ib_background = image_background.at<cv::Vec3b>(v, u)[0];
			int Ir = image.at<cv::Vec3b>(v, u)[2];
			int Ig = image.at<cv::Vec3b>(v, u)[1];
			int Ib = image.at<cv::Vec3b>(v, u)[0];

			int d_background = (int)depth_background.at<unsigned short>(v, u);
			int d = (int)depth.at<unsigned short>(v, u);

			//if ((d_background - d > 50) && abs(Ir - Ir_background + Ig - Ig_background + Ib - Ib_background) > 60) {
			if ((d_background - d > 50)) {
				output[v*width + u] = 100;
			}
			else output[v*width + u] = 255;
		}
	}

	cv::Mat output_image(height, width, CV_8UC1, output);
	cv::cvtColor(output_image, output_image, CV_GRAY2BGR);
	cv::imwrite(outputAnno, output_image, compression_params);
	std::cout << "success" << std::endl;
	delete[] output;
}

// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];
unsigned int getColor( const unsigned char * c ){
	return c[0] + 256*c[1] + 256*256*c[2];
}
void putColor( unsigned char * c, unsigned int cc ){
	c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize( const short * map, int W, int H ){
	unsigned char * r = new unsigned char[ W*H*3 ];
	for( int k=0; k<W*H; k++ ){
		int c = colors[ map[k] ];
		putColor( r+3*k, c );
	}
	return r;
}

// Certainty that the groundtruth is correct
const float GT_PROB = 0.9;

// Simple classifier that is 50% certain that the annotation is correct
float * classify( const unsigned char * im, int W, int H, int M ){
	const float u_energy = -log( 1.0f / M );
	const float n_energy = -log( (1.0f - GT_PROB) / (M-1) );
	const float p_energy = -log( GT_PROB );
	float * res = new float[W*H*M];
	for( int k=0; k<W*H; k++ ){
		// Map the color to a label
		int c = getColor( im + 3*k );
		int i;
		for( i=0;i<nColors && c!=colors[i]; i++ );
		if (c && i==nColors){//c is not stored yet
			if (i<M)
				colors[nColors++] = c;
			else
				c=0;
		}
		
		// Set the energy
		float * r = res + k*M;
		if (c){// c is mapped to i-th label
			for( int j=0; j<M; j++ )
				r[j] = n_energy;
			r[i] = p_energy;
		}
		else{
			for( int j=0; j<M; j++ )
				r[j] = u_energy;
		}
	}
	return res;
}

int main( int argc, char* argv[]){
	if (argc<8){
		printf("Usage: %s backgroundRGB backgroundDepth imageRGB imageDepth ppmImage ppmAnnotation output\n", argv[0] );
		return 1;
	}

	preprocessing(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
	// Number of labels
	const int M = 2;
	// Load the color image and some crude annotations (which are used in a simple classifier)
	int W, H, GW, GH;
	//W：width of image, H: height of image
	//GW: width of annotation image, GH: height of annotation image
	unsigned char * im = readPPM(argv[5], W, H);
	if (!im){
		printf("Failed to load image!\n");
		return 1;
	}
	unsigned char * anno = readPPM(argv[6], GW, GH);
	if (!anno){
		printf("Failed to load annotations!\n");
		return 1;
	}
	if (W!=GW || H!=GH){
		printf("Annotation size doesn't match image!\n");
		return 1;
	}
	
	/////////// Put your own unary classifier here! ///////////
	float * unary = classify( anno, W, H, M );
	///////////////////////////////////////////////////////////
	
	// Setup the CRF model
	DenseCRF2D crf(W, H, M);
	// Specify the unary potential as an array of size W*H*(#classes)
	// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
	crf.setUnaryEnergy( unary );
	// add a color independent term (feature = pixel location 0..W-1, 0..H-1)
	// x_stddev = 3 ********Standard deviation*********
	// y_stddev = 3
	// weight = 3
	crf.addPairwiseGaussian( 3, 3, 5 );
	// add a color dependent term (feature = xyrgb)
	// x_stddev = 60
	// y_stddev = 60
	// r_stddev = g_stddev = b_stddev = 20
	// weight = 10
	crf.addPairwiseBilateral( 60, 60, 20, 20, 20, im, 5 );
	
	// Do map inference
	short * map = new short[W*H];
	crf.map(10, map);// MAP
	
	// Store the result
	unsigned char *res = colorize( map, W, H );
	writePPM( argv[7], W, H, res );
	
	delete[] im;
	delete[] anno;
	delete[] res;
	delete[] map;
	delete[] unary;
}
