#include<iostream>
#include<cstring>
#include<chrono>
#include <sycl/sycl.hpp>          //# sycl namespace
#include "oneapi/mkl/blas.hpp"  //# oneMKL DPC++ interface for BLAS functions

// using namespace std;
using namespace sycl;
namespace mkl = oneapi::mkl;  //# shorten mkl namespace
extern "C" {
    double kn2row(const unsigned char* image_data, int rows, int cols, const int channels, const int channels_out,
        const unsigned char* kernel_data,  int k, int stride, int padding, int opDim, 
        const unsigned char* output_data){
        
        //Copying the image and kernel data from python
        float* image = new float[channels * rows * cols];
        std::memcpy(image, image_data, (channels * rows * cols) * sizeof(float));

        float* kernel = new float[channels * channels_out * k * k];
        std::memcpy(kernel, kernel_data, (channels * channels_out * k * k) * sizeof(float));

        //Starting the timer
        auto start = std::chrono::high_resolution_clock::now();
        
        //Creating the input image matrix;
        float** kn2row_input_array = new float*[channels];
        int input_H = channels;
        int input_W = ((rows + 2 * padding) * (cols + 2 * padding));
        for(int c = 0; c<channels; c++){
            
            int inpItr = 0;
            kn2row_input_array[c] = new float[input_W];
            
            for(int i=0; i<padding*(cols + 2 * padding); i++){
                kn2row_input_array[c][i] = 0;
            }

            for(int i = padding; i<rows+padding; i++){
                for(int p=0; p<padding; p++){
                    kn2row_input_array[c][i*(cols + 2 * padding) + p] = 0;
                }
                for(int j=padding; j<cols+padding; j++){
                    kn2row_input_array[c][i * (cols + 2 * padding)+j] = image[c * (rows * cols) + inpItr];
                    inpItr++;
                }
                for(int p=cols+padding; p<(cols + 2 * padding); p++){
                    kn2row_input_array[c][i*(cols + 2 * padding) + p] = 0;
                }
            }

            for(int i=((cols + 2 * padding) * (rows + padding)); i<((cols + 2 * padding) * (rows + 2 * padding)); i++){
                kn2row_input_array[c][i] = 0;
            }

        }

        //Creating the kernel matrix
        float* kn2row_kernel_array[k * k * channels_out];
        for(int c=0; c<channels_out; c++){
            for(int i=0; i<k*k; i++){
                kn2row_kernel_array[(c * k * k) + i] = new float[channels];
                for(int j=0; j<channels; j++){
                    kn2row_kernel_array[(c * k * k) + i][j] = kernel[(channels * c * k * k) + (j * k * k) + i];
                }
            }
        }
        
        //Multiplying the kernel and input matrices
        queue q(property::queue::enable_profiling{});
        
        float *matrixA = sycl::malloc_shared<float>(k * k * channels_out * channels, q);
        float *matrixB = sycl::malloc_shared<float>(channels * input_W, q);
        float *matrixC = sycl::malloc_shared<float>(k * k * channels_out * input_W, q);
        
        for(int i=0; i<k * k * channels_out; i++){
            for(int j=0; j<channels; j++){
                matrixA[i * channels + j] = kn2row_kernel_array[i][j];
            }
        }
        
        for(int i=0; i<channels; i++){
            for(int j=0; j<input_W; j++){
                matrixB[i * input_W + j] = kn2row_input_array[i][j];
            }
        }
        
        // Clearing the constructed input matrices as they are not needed anymore
        delete[] image;
        delete[] kernel;
        for (int i = 0; i < input_H; ++i) {
            delete[] kn2row_input_array[i];
        }
        delete[] kn2row_input_array;
        
        for(int i = 0; i < (k * k * channels_out); ++i){
            delete[] kn2row_kernel_array[i];
        }
        
        sycl::event gemm_done;
        std::vector<sycl::event> gemm_dependencies;
        
        float alpha = 1.0, beta = 1.0;
    
        mkl::transpose transA = mkl::transpose::nontrans;
        mkl::transpose transB = mkl::transpose::nontrans;
        int ldA = channels, ldB = input_W, ldC = input_W;

        // Define a DPC++ kernel for matrix multiplication
        gemm_done = mkl::blas::row_major::gemm(q, transA, transB, k * k * channels_out, input_W, channels, alpha, matrixA, ldA, matrixB, ldB, beta, matrixC, ldC, gemm_dependencies);
        
        //Copying the multiplication output to 2d matrix for shift and add operation
        float *productMatrix[channels_out * k * k];
        for (int i=0; i <(k * k * channels_out); i++ ){
            productMatrix[i] = new float[input_W];
            for(int j=0; j<input_W; j++ ){
                productMatrix[i][j] = matrixC[i * (input_W) + j];
            }
        }

        //Performing shift and add
        float shiftAdd[channels_out][opDim][opDim];

        for(int c=0;c<channels_out;c++){
            for(int o=0, sr=0; o<opDim, sr<rows; o++, sr+=stride){
                for(int i=0, sc=0; i<opDim, sc<cols; i++, sc+=stride){
                    for(int l=0; l<k; l++){
                        for(int a=0;a<k;a++){
                            shiftAdd[c][o][i] = shiftAdd[c][o][i] + productMatrix[sr + a + k *l + c * k * k][sc + cols*l + a + o*rows];
                        }
                    }
                }
            }
        }

        //Stopping the timer
        auto stop = std::chrono::high_resolution_clock::now();

        //Calculating the time taken
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double executionTimeMicroseconds = duration.count();

        //Some operations for copying the final output to the python output_data array
        float reshape[channels_out][opDim * opDim];
        for(int c=0; c<channels_out; c++){
            for(int i=0; i<opDim; i++){
                for(int j=0; j<opDim; j++){
                    reshape[c][i*opDim + j] = shiftAdd[c][i][j];
                }
            }
        }
        
        //Copying our output array to the python output_data pointer.
        for (int i = 0; i < channels_out; ++i) {
            memcpy((void*)(output_data + i * (opDim * opDim) * sizeof(float)), (void*)reshape[i], (opDim * opDim) * sizeof(float));
        }

        //Returning the execution time
        return executionTimeMicroseconds;
        
    }
}