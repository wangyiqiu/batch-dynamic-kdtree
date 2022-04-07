#pragma once

#include "dataset/uniform.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include <random>

//using namespace batchKdTree;

// --- Taken from parlaylib ---
// Use this macro to avoid accidentally timing the destructors
// of the output produced by algorithms that return data
//
// The expression e is evaluated as if written in the context
// auto result_ = (e);
//
#define RUN_AND_CLEAR(e) \
  {                      \
    auto result_ = (e);  \
    state.PauseTiming(); \
  }                      \
  state.ResumeTiming();

// Define the benchmarks
#define BENCH(NAME, ...) \
  BENCHMARK_TEMPLATE(bench_##NAME, ##__VA_ARGS__)->UseRealTime()->Unit(benchmark::kMillisecond)

// --- Data loading ---
template <int dim>
void AddNoise(parlay::sequence<batchKdTree::point<dim>>& points) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1e-6);

  for (size_t i = 0; i < points.size(); i++) {
    for (int d = 0; d < dim; d++) {
      points[i][d] += distribution(generator);
    }
  }
}

template <int dim, bool add_noise = false>
auto LoadFile(const char* filePath) {
  [[maybe_unused]] auto read_dim = readDimensionFromFile(filePath);
  assert(read_dim == dim);
  auto ret = readPointsFromFile<batchKdTree::point<dim>>(filePath);
  if (add_noise) AddNoise<dim>(ret);
  return ret;
}

// Global declaration of data set fields

parlay::sequence<batchKdTree::point<2>> us2d_1k;
parlay::sequence<batchKdTree::point<2>> us2d_10k;
parlay::sequence<batchKdTree::point<2>> us2d_1m;
parlay::sequence<batchKdTree::point<2>> us2d_10m;

auto UniformSphere2D_1K() {
  if (us2d_1k.size() == 0)
    us2d_1k = pargeo::uniformInPolyPoints<2, batchKdTree::point<2>>(1000, 0);
  return us2d_1k;
}
auto UniformSphere2D_10K() {
  if (us2d_10k.size() == 0)
    us2d_10k = pargeo::uniformInPolyPoints<2, batchKdTree::point<2>>(10000, 0);
  return us2d_10k;
}
auto UniformSphere2D_1M() {
  if (us2d_1m.size() == 0)
    us2d_1m = pargeo::uniformInPolyPoints<2, batchKdTree::point<2>>(1000000, 0);
  return us2d_1m;
}
auto UniformSphere2D_10M() {
  if (us2d_10m.size() == 0)
    us2d_10m = pargeo::uniformInPolyPoints<2, batchKdTree::point<2>>(10000000, 0);
  return us2d_10m;
}
auto VisualVar5D_100K() { 
  throw std::runtime_error("VisualVar5D_100K not available");
  return LoadFile<5>("datasets/5D_VisualVar_100K.pbbs"); }
auto VisualVar5D_1M() {
  throw std::runtime_error("VisualVar5D_1M not available");
  return LoadFile<5>("datasets/5D_VisualVar_1M.pbbs"); }

// Benchmark Datasets

parlay::sequence<batchKdTree::point<2>> uf2d_10m;
parlay::sequence<batchKdTree::point<5>> uf5d_10m;
parlay::sequence<batchKdTree::point<7>> uf7d_10m;

auto UniformFill2D_10M() {
  if (uf2d_10m.size() == 0)
    uf2d_10m = pargeo::uniformInPolyPoints<2, batchKdTree::point<2>>(10000000, 1);
  return uf2d_10m;
}
auto UniformFill5D_10M() {
  if (uf5d_10m.size() == 0)
    uf5d_10m = pargeo::uniformInPolyPoints<5, batchKdTree::point<5>>(10000000, 1);
  return uf5d_10m;
}
auto UniformFill7D_10M() {
  if (uf7d_10m.size() == 0)
    uf7d_10m = pargeo::uniformInPolyPoints<7, batchKdTree::point<7>>(10000000, 1);
  return uf7d_10m;
}
auto VisualVar2D_10M() {
  throw std::runtime_error("VisualVar2D_10M not available");
  return LoadFile<2, true>("datasets/2D_VisualVar_10M.pbbs"); }
auto VisualVar5D_10M() {
  throw std::runtime_error("VisualVar5D_10M not available");
  return LoadFile<5, true>("datasets/5D_VisualVar_10M.pbbs"); }
auto VisualVar7D_10M() {
  throw std::runtime_error("VisualVar7D_10M not available");
  return LoadFile<7, true>("datasets/7D_VisualVar_10M.pbbs"); }
auto GeoLife3D_24M() {
  throw std::runtime_error("GeoLife3D_24M not available");
  return LoadFile<3, true>("datasets/3D_GeoLife_24M.pbbs"); }
auto HouseHold7D_2M() {
  throw std::runtime_error("HouseHold7D_2M not available");
  return LoadFile<7, true>("datasets/7D_HouseHold_2M.pbbs"); }
auto HT10D_1M() {
  throw std::runtime_error("HT10D_1M not available");
  return LoadFile<10, true>("datasets/HT.pbbs"); }
auto CHEM16D_4M() {
  throw std::runtime_error("CHEM16D_4M not available");
  return LoadFile<16, true>("datasets/CHEM.pbbs"); }

enum DSType {
  DS_UNIFORM_FILL,    // 0
  DS_UNIFORM_SPHERE,  // 1
  DS_VISUAL_VAR,      // 2
  DS_GEO_LIFE,        // 3
  DS_HOUSE_HOLD,      // 4
  DS_HT,              // 5
  DS_CHEM             // 6
};
template <int dim>
auto BenchmarkDS(__attribute__((unused)) int size,
                 __attribute__((unused)) DSType ds_type,
                 bool shuffle = true) {
  std::stringstream ss;
  ss << "Invalid dim=" << dim << " to BenchmarkDS";
  throw std::runtime_error(ss.str());
}

template <>
auto BenchmarkDS<2>(int size, DSType ds_type, bool shuffle) {
  parlay::sequence<batchKdTree::point<2>> ret;
  switch (ds_type) {
    case DS_UNIFORM_FILL: {
      ret = UniformFill2D_10M();
      break;
    }
    case DS_UNIFORM_SPHERE: {
      if (size == 1'000)
        ret = UniformSphere2D_1K();
      else if (size == 10'000)
        ret = UniformSphere2D_10K();
      else if (size == 1'000'000)
        ret = UniformSphere2D_1M();
      else if (size == 10'000'000)
        ret = UniformSphere2D_10M();
      else
        throw std::runtime_error("Invalid size to BenchmarkDS<2>(DS_UNIFORM_SPHERE)");
      break;
    }
    case DS_VISUAL_VAR: {
      ret = VisualVar2D_10M();
      break;
    }
    default:
      throw std::runtime_error("Invalid type to BenchmarkDS<2>");
  }

  if (shuffle)
    return random_shuffle(ret);
  else
    return ret;
}

template <>
auto BenchmarkDS<3>(__attribute__((unused)) int size, DSType ds_type, bool shuffle) {
  parlay::sequence<batchKdTree::point<3>> ret;
  switch (ds_type) {
    case DS_GEO_LIFE: {
      ret = GeoLife3D_24M();
      break;
    }
    default:
      throw std::runtime_error("Invalid type to BenchmarkDS<3>");
  }

  if (shuffle)
    return random_shuffle(ret);
  else
    return ret;
}

template <>
auto BenchmarkDS<5>(int size, DSType ds_type, bool shuffle) {
  parlay::sequence<batchKdTree::point<5>> ret;
  switch (ds_type) {
    case DS_UNIFORM_FILL: {
      ret = UniformFill5D_10M();
      break;
    }
    case DS_VISUAL_VAR: {
      if (size == 100'000)
        ret = VisualVar5D_100K();
      else if (size == 1'000'000)
        ret = VisualVar5D_1M();
      else if (size == 10'000'000)
        ret = VisualVar5D_10M();
      else
        throw std::runtime_error("Invalid size to BenchmarkDS<5>(DS_VISUAL_VAR)");
      break;
    }
    default:
      throw std::runtime_error("Invalid type to BenchmarkDS<5>");
  }

  if (shuffle)
    return random_shuffle(ret);
  else
    return ret;
}

template <>
auto BenchmarkDS<7>(__attribute__((unused)) int size, DSType ds_type, bool shuffle) {
  parlay::sequence<batchKdTree::point<7>> ret;
  switch (ds_type) {
    case DS_UNIFORM_FILL: {
      ret = UniformFill7D_10M();
      break;
    }
    case DS_VISUAL_VAR: {
      ret = VisualVar7D_10M();
      break;
    }
    case DS_HOUSE_HOLD: {
      ret = HouseHold7D_2M();
      break;
    }
    default:
      throw std::runtime_error("Invalid type to BenchmarkDS<7>");
  }

  if (shuffle)
    return random_shuffle(ret);
  else
    return ret;
}

template <>
auto BenchmarkDS<10>(__attribute__((unused)) int size, DSType ds_type, bool shuffle) {
  parlay::sequence<batchKdTree::point<10>> ret;
  switch (ds_type) {
    case DS_HT: {
      ret = HT10D_1M();
      break;
    }
    default:
      throw std::runtime_error("Invalid type to BenchmarkDS<10>");
  }

  if (shuffle)
    return random_shuffle(ret);
  else
    return ret;
}

template <>
auto BenchmarkDS<16>(__attribute__((unused)) int size, DSType ds_type, bool shuffle) {
  parlay::sequence<batchKdTree::point<16>> ret;
  switch (ds_type) {
    case DS_CHEM: {
      ret = CHEM16D_4M();
      break;
    }
    default:
      throw std::runtime_error("Invalid type to BenchmarkDS<16>");
  }

  if (shuffle)
    return random_shuffle(ret);
  else
    return ret;
}

// types
constexpr bool parallel = true;
constexpr bool coarsen = true;

template <int dim>
using COTree_t = batchKdTree::CO_KdTree<dim, batchKdTree::point<dim>, parallel, coarsen>;

template <int dim>
using BHLTree_t = batchKdTree::BHL_KdTree<dim, batchKdTree::point<dim>, parallel, coarsen>;

constexpr int NUM_TREES = 21;
constexpr int BUFFER_LOG2_SIZE = 10;
template <int dim>
using LogTree_t = batchKdTree::LogTree<NUM_TREES, BUFFER_LOG2_SIZE, dim, batchKdTree::point<dim>, parallel, coarsen>;
