// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  r.origin = eye;

  glm::vec3 right = glm::normalize(glm::cross(view, up));
  //TODO: use glm::tan and glm::radians
  float d = 0.5f * resolution.y / tan(fov.y*(PI/180.f)); // distance from the eye to the image plane
  r.direction = glm::normalize(view*d + right*(0.5f*resolution.x - x) + up*(0.5f*resolution.y - y));
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms,
							const material* materials, int numOfMaterials,
							const int* lightIndices, int numOfLights){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y)){
		// colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
		colors[index] = glm::vec3(0.f, 0.f, 0.f);
		ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		
		glm::vec3 intersectionPoint, normal;
		float intersectionDistance;
		int intersectionGeomInd = findClosestIntersection(geoms, numberOfGeoms, r,
			&intersectionPoint, &normal, &intersectionDistance);

		if (intersectionGeomInd != -1) { // found the closest front object
			const material& objectMaterial = materials[geoms[intersectionGeomInd].materialid];
			glm::vec3 diffuseColor = objectMaterial.color;
			glm::vec3 specularColor = objectMaterial.specularColor;

			if (objectMaterial.emittance > EPSILON) {
				// object to be rendered is a light source
				colors[index] = diffuseColor;
				return;
			}

			/* Phong Illumination Model */
			/* ka*diffuse_color + kd*diffuse_color*(N*L) + ks*specular_color*(N*H)^exp_n 
			 * N(normal) : unit vector, the direction of the surface normal at the intersection 
			 * L(ligtDirection) : unit vector, the direction of the vector to the light source from the intersection 
			 * H : unit vector, the direction that is halfway between the direction to the light and the direction to the viewer */
			glm::vec3 diffuse_sum(0.f, 0.f, 0.f);
			glm::vec3 specular_sum(0.f, 0.f, 0.f);
			for (int i = 0; i < numOfLights; i++) { // for each light source
				int lightInd = lightIndices[i];
				const staticGeom& light = geoms[lightInd];
				glm::vec3 lightCenter = multiplyMV(light.transform, glm::vec4(0.f, 0.f, 0.f, 1.0f));
				glm::vec3 lightDirection = glm::normalize(lightCenter - intersectionPoint);

				// check occulusion for shadow
				// NOTE: move the intersection point toward each light a little bit to avoid numerical error
				ray lightRay; lightRay.origin = intersectionPoint + lightDirection*float(RAY_BIAS_AMOUNT); lightRay.direction = lightDirection;
				int obstacleGeomInd = findClosestIntersection(geoms, numberOfGeoms, lightRay);
				if (obstacleGeomInd != lightInd) {
					continue;
				}
				
				glm::vec3 V = glm::normalize(-r.direction);
				glm::vec3 H = glm::normalize(lightDirection + V);

				const material& lightMaterial = materials[light.materialid];
				glm::vec3 lightColor = lightMaterial.color;
				diffuse_sum += lightColor * max(0.f, glm::dot(normal, lightDirection));

				if (glm::dot(normal, H) > EPSILON) {
					specular_sum += lightColor * (glm::pow(glm::dot(normal, H), objectMaterial.specularExponent));
				}
			}

			colors[index] = glm::clamp(0.3f*diffuseColor + diffuseColor*diffuse_sum + specularColor*specular_sum, 0.f, 1.f); 
			//colors[index] = glm::clamp(0.3f*diffuseColor + diffuseColor*diffuse_sum, 0.f, 1.f); 
		}
	}
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and indices for lights and send them to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  int* lightIndices = new int[numberOfGeoms];
  int numOfLights = 0;
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;

	if (materials[newStaticGeom.materialid].emittance > EPSILON) {
		lightIndices[numOfLights++] = i;
	}
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  int* cudaLightIndices = NULL;
  cudaMalloc((void**)&cudaLightIndices, numOfLights*sizeof(int));
  cudaMemcpy( cudaLightIndices, lightIndices, numOfLights*sizeof(int), cudaMemcpyHostToDevice);

  // package materials and send to GPU
  material* cudaMaterials = NULL;
  cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy(cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, 
	  cudageoms, numberOfGeoms, cudaMaterials, numberOfMaterials, cudaLightIndices, numOfLights);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudaMaterials );
  cudaFree( cudaLightIndices );
  delete geomList;
  delete lightIndices;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
