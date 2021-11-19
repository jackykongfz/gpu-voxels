// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-07-09
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_HPP_INCLUDED

#include "ProbVoxelMap.h"
#include <gpu_voxels/voxelmap/TemplateVoxelMap.hpp>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.hpp>
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/voxel/ProbabilisticVoxel.hpp>
#include <vector>

namespace gpu_voxels {
namespace voxelmap {

ProbVoxelMap::ProbVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dim, voxel_side_length, map_type)
{
  map_res = voxel_side_length;
  // int map_allsize = ceil(dim.x/voxel_side_length) + ceil(dim.y/voxel_side_length) + ceil(dim.z/voxel_side_length);
  int map_allsize = dim.x * dim.y * dim.z;
  // hash_map = (int*)malloc(map_allsize);
  cudaMallocManaged((void**)&hash_map, map_allsize);
}

ProbVoxelMap::ProbVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dev_data, dim, voxel_side_length, map_type)
{
  map_res = voxel_side_length;
  int map_allsize = ceil(dim.x/voxel_side_length) + ceil(dim.y/voxel_side_length) + ceil(dim.z/voxel_side_length);
  // hash_map = (int*)malloc(map_allsize);
  cudaMallocManaged((void**)&hash_map, map_allsize);
}

ProbVoxelMap::~ProbVoxelMap()
{

}

template<std::size_t length>
void ProbVoxelMap::insertSensorData(const PointCloud &global_points, const Vector3f &sensor_pose, const bool enable_raycasting,
                                    const bool cut_real_robot, const BitVoxelMeaning robot_voxel_meaning,
                                    BitVoxel<length>* robot_map)
{
  lock_guard guard(this->m_mutex);

  computeLinearLoad(global_points.getPointCloudSize(), &m_blocks,
                           &m_threads);

  // int* current_voxel_count=new int [global_points.getPointCloudSize()];
    int nBytes = global_points.getPointCloudSize() * sizeof(int);
    // apply host memory
    int *current_voxel_count, *current_voxel_count_cuda;
    current_voxel_count = (int*)malloc(nBytes);
    cudaMalloc((void**)&current_voxel_count_cuda, nBytes);



  if (enable_raycasting)
  {
    kernelInsertSensorData<<<m_blocks, m_threads>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, sensor_pose,
        global_points.getConstDevicePointer(), global_points.getPointCloudSize(), cut_real_robot, robot_map, robot_voxel_meaning, RayCaster(), current_voxel_count_cuda, hash_map);
    CHECK_CUDA_ERROR();
  }
  else
  {
    kernelInsertSensorData<<<m_blocks, m_threads>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, sensor_pose,
        global_points.getConstDevicePointer(), global_points.getPointCloudSize(), cut_real_robot, robot_map, robot_voxel_meaning, DummyRayCaster(), current_voxel_count_cuda, hash_map);
    CHECK_CUDA_ERROR();
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  cudaMemcpy((void*)current_voxel_count, (void*)current_voxel_count_cuda, nBytes, cudaMemcpyDeviceToHost);

  for(int i=0;i<global_points.getPointCloudSize();i++)
  {
    all_voxel_count = all_voxel_count + current_voxel_count[i];
  }

  // all_voxel_count = all_voxel_count + current_voxel_count;
}

bool ProbVoxelMap::insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud *robot_links,
                                                              const std::vector<BitVoxelMeaning>& voxel_meanings,
                                                              const std::vector<BitVector<BIT_VECTOR_LENGTH> >& collision_masks,
                                                              BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
{
  LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return true;
}

void ProbVoxelMap::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  if(voxel_meaning != eBVM_OCCUPIED)
     LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
    this->clearMap();
}

//Collsion Interface Implementations

size_t ProbVoxelMap::collideWith(const BitVectorVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  return collisionCheckWithCounterRelativeTransform((TemplateVoxelMap*)map, collider, offset); //does the locking

}

size_t ProbVoxelMap::collideWith(const ProbVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  return collisionCheckWithCounterRelativeTransform((TemplateVoxelMap*)map, collider, offset); //does the locking
}

int32_t ProbVoxelMap::get_vovel_count()
{
  return all_voxel_count;
}

float ProbVoxelMap::get_explored_volume()
{
  return all_voxel_count*map_res*map_res*map_res;
}

} // end of namespace
} // end of namespace

#endif
