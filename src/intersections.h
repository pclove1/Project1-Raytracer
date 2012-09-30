// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

__host__ __device__ void mySwap(float &a, float &b) // mySwap gives me a compilation error
{
  float temp = a;
  a = b;
  b = temp;
}


//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ bool boxIntersectionTest(staticGeom box, ray r);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ bool sphereIntersectionTest(staticGeom sphere, ray r);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);
__host__ __device__ bool findClosestIntersection(const staticGeom* geoms, int numOfGeoms, const ray& r, 
												glm::vec3* closestIntersection, glm::vec3* closestIntersectionNormal, 
												float* closestDistance, int* closestMaterialInd);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b))<EPSILON){
        return true;
    }else{
        return false;
    }
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors. 
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	float minBound = -0.5f;
	float maxBound = 0.5f;

	// transform the ray r to the object-space from the world-space
	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
	ray rt; rt.origin = ro; rt.direction = rd;

	//TODO: find different constants for CUDA not for CPU
	float t_near = -FLT_MAX;
	float t_far = FLT_MAX;

	// YZ plane(X is normal)
	if (fabs(rt.direction.x) < EPSILON // rt is parallel to YZ plane
		&& (rt.origin.x < minBound || rt.origin.x > maxBound)) {	
		return -1;
	} 
	float dDiv = 1 / rt.direction.x;
	float t1 = (minBound - rt.origin.x)*dDiv;
	float t2 = (maxBound - rt.origin.x)*dDiv;
	if (t1 > t2) mySwap(t1, t2);
	if (t1 > t_near) t_near = t1;
	if (t2 < t_far) t_far = t2;	
	if (t_far < 0.f) return -1; // cube is behind

	// ZX plane(Y is normal)
	if (fabs(rt.direction.y) < EPSILON // rt is parallel to ZX plane
		&& (rt.origin.y < minBound || rt.origin.y > maxBound)) {	
		return -1;
	} 
	dDiv = 1 / rt.direction.y;
	t1 = (minBound - rt.origin.y)*dDiv;
	t2 = (maxBound - rt.origin.y)*dDiv;
	if (t1 > t2) mySwap(t1, t2);
	if (t1 > t_near) t_near = t1;
	if (t2 < t_far) t_far = t2;	
	if (t_near > t_far) return -1; // cube is missed
	if (t_far < 0.f) return -1; // cube is behind

	// XY plane(Z is normal)
	if (fabs(rt.direction.z) < EPSILON // rt is parallel to XY plane
		&& (rt.origin.z < minBound || rt.origin.z > maxBound)) {	
		return -1;
	} 
	dDiv = 1 / rt.direction.z;
	t1 = (minBound - rt.origin.z)*dDiv;
	t2 = (maxBound - rt.origin.z)*dDiv;
	if (t1 > t2) mySwap(t1, t2);
	if (t1 > t_near) t_near = t1;
	if (t2 < t_far) t_far = t2;	
	if (t_near > t_far) return -1; // cube is missed
	if (t_far < 0.f) return -1; // cube is behind

	// compute the final t
	float t = t_near;
	if (t_near < 0.f) t = t_far;

	// compute the real intersection point
	glm::vec3 intersectionPointInObjectSpace = getPointOnRay(rt, t);
	intersectionPoint = multiplyMV(box.transform, glm::vec4(intersectionPointInObjectSpace, 1.0f));

	glm::vec3 normalInObjectSpace;
	if (fabs(intersectionPointInObjectSpace.x - 0.5f) < EPSILON) {
		normalInObjectSpace = glm::vec3(1.f, 0.f, 0.f);
	} else if (fabs(intersectionPointInObjectSpace.x + 0.5f) < EPSILON) {
		normalInObjectSpace = glm::vec3(-1.f, 0.f, 0.f);
	} else if (fabs(intersectionPointInObjectSpace.y - 0.5f) < EPSILON) {
		normalInObjectSpace = glm::vec3(0.f, 1.f, 0.f);
	} else if (fabs(intersectionPointInObjectSpace.y + 0.5f) < EPSILON) {
		normalInObjectSpace = glm::vec3(0.f, -1.f, 0.f);
	} else if (fabs(intersectionPointInObjectSpace.z - 0.5f) < EPSILON) {
		normalInObjectSpace = glm::vec3(0.f, 0.f, 1.f);
	} else if (fabs(intersectionPointInObjectSpace.z + 0.5f) < EPSILON) {
		normalInObjectSpace = glm::vec3(0.f, 0.f, -1.f);
	} 
	// don't forget to transform normal in object space to real-world space
	normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normalInObjectSpace, 1.0f)));
        
	return glm::length(r.origin - intersectionPoint);    
}

__host__ __device__ bool boxIntersectionTest(staticGeom box, ray r) {
	float minBound = -0.5f;
	float maxBound = 0.5f;

	// transform the ray r to the object-space from the world-space
	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
	ray rt; rt.origin = ro; rt.direction = rd;

	//TODO: find different constants for CUDA not for CPU
	float t_near = -FLT_MAX;
	float t_far = FLT_MAX;

	// YZ plane(X is normal)
	if (fabs(rt.direction.x) < EPSILON // rt is parallel to YZ plane
		&& (rt.origin.x < minBound || rt.origin.x > maxBound)) {	
		return false;
	} 
	float dDiv = 1 / rt.direction.x;
	float t1 = (minBound - rt.origin.x)*dDiv;
	float t2 = (maxBound - rt.origin.x)*dDiv;
	if (t1 > t2) mySwap(t1, t2);
	if (t1 > t_near) t_near = t1;
	if (t2 < t_far) t_far = t2;	
	if (t_far < 0.f) return false; // cube is behind

	// ZX plane(Y is normal)
	if (fabs(rt.direction.y) < EPSILON // rt is parallel to ZX plane
		&& (rt.origin.y < minBound || rt.origin.y > maxBound)) {	
		return false;
	} 
	dDiv = 1 / rt.direction.y;
	t1 = (minBound - rt.origin.y)*dDiv;
	t2 = (maxBound - rt.origin.y)*dDiv;
	if (t1 > t2) mySwap(t1, t2);
	if (t1 > t_near) t_near = t1;
	if (t2 < t_far) t_far = t2;	
	if (t_near > t_far) return false; // cube is missed
	if (t_far < 0.f) return false; // cube is behind

	// XY plane(Z is normal)
	if (fabs(rt.direction.z) < EPSILON // rt is parallel to XY plane
		&& (rt.origin.z < minBound || rt.origin.z > maxBound)) {	
		return false;
	} 
	dDiv = 1 / rt.direction.z;
	t1 = (minBound - rt.origin.z)*dDiv;
	t2 = (maxBound - rt.origin.z)*dDiv;
	if (t1 > t2) mySwap(t1, t2);
	if (t1 > t_near) t_near = t1;
	if (t2 < t_far) t_far = t2;	
	if (t_near > t_far) return false; // cube is missed
	if (t_far < 0.f) return false; // cube is behind

	return true;; 
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }     

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);   
        
  return glm::length(r.origin - realIntersectionPoint);
}

__host__ __device__ bool sphereIntersectionTest(staticGeom sphere, ray r) {
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return false;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  if (t1 < 0 && t2 < 0) {
      return false;
  } 

  return true;
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

//LOOK: Example for generating a random point on an object using thrust. 
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    //get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f;   //x-y face
    float side2 = radii.z * radii.y * 4.0f;   //y-z face
    float side3 = radii.x * radii.z* 4.0f;   //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    //pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        //x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        //x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        //y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        //y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        //x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        //x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

  return glm::vec3(0,0,0);
}

__host__ __device__ bool findClosestIntersection(const staticGeom* geoms, int numOfGeoms, const ray& r, 
												glm::vec3* closestIntersection, glm::vec3* closestIntersectionNormal, 
												float* closestDistance, int* closestMaterialInd) {
	glm::vec3 intersectionPoint, normal;
	float intersectionDistance;
	*closestDistance = FLT_MAX;
	*closestMaterialInd = -1;

	for (int i = 0; i < numOfGeoms; i++) { // for each object
		if (geoms[i].type == SPHERE) {
			intersectionDistance = sphereIntersectionTest(geoms[i], r, intersectionPoint, normal);
		} else if (geoms[i].type == CUBE) {
			intersectionDistance = boxIntersectionTest(geoms[i], r, intersectionPoint, normal);
		} else { // not-supported object type
			continue;
		}

		if (intersectionDistance < 0.f) { // object is missed
			continue; 
		} else if (intersectionDistance < *closestDistance) { // closer is found
			*closestDistance = intersectionDistance;
			*closestMaterialInd = geoms[i].materialid;
			*closestIntersection = intersectionPoint;
			*closestIntersectionNormal = normal;
		}
	}	

	return (*closestMaterialInd != -1);
}

#endif