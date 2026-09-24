// In-process FLINT side of the lattica Gram-LLL comparison benchmark.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_lll.h>
#include <flint/fmpz_mat.h>

namespace {

constexpr int dimensions[] = {8, 16, 24};
constexpr std::size_t lll_cases = 16;
constexpr std::size_t samples = 11;
volatile std::int64_t result_sink = 0;

struct Rng {
  std::uint64_t state;

  std::uint64_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }

  std::size_t index(std::size_t bound) {
    return static_cast<std::size_t>(next() % bound);
  }
};

using Rows = std::vector<std::int64_t>;

Rows canonical_basis(std::size_t dimension) {
  Rows basis(dimension * dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    basis[row * dimension + row] = 2;
    if (row > 0) {
      basis[row * dimension + row - 1] = 1;
    }
    if (row > 1) {
      basis[row * dimension + row - 2] = -1;
    }
  }
  return basis;
}

Rows skew_basis(std::size_t dimension, std::size_t benchmark_case) {
  auto basis = canonical_basis(dimension);
  Rng rng{0xd1b54a32d192ed03ULL ^ static_cast<std::uint64_t>(dimension) ^
          (static_cast<std::uint64_t>(benchmark_case) << 32)};
  for (std::size_t step = 0; step < 2 * dimension; ++step) {
    const auto destination = rng.index(dimension);
    auto source = rng.index(dimension - 1);
    if (source >= destination) {
      ++source;
    }
    const std::int64_t sign = (rng.next() & 1) == 0 ? -1 : 1;
    const auto destination_start = destination * dimension;
    const auto source_start = source * dimension;
    bool acceptable = true;
    for (std::size_t column = 0; column < dimension; ++column) {
      const auto candidate = basis[destination_start + column] +
                             sign * basis[source_start + column];
      acceptable = acceptable && std::abs(candidate) <= 256;
    }
    if (acceptable) {
      for (std::size_t column = 0; column < dimension; ++column) {
        basis[destination_start + column] +=
            sign * basis[source_start + column];
      }
    }
  }
  return basis;
}

Rows gram(const Rows &basis, std::size_t dimension) {
  Rows result(dimension * dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = row; column < dimension; ++column) {
      std::int64_t value = 0;
      for (std::size_t k = 0; k < dimension; ++k) {
        value += basis[row * dimension + k] * basis[column * dimension + k];
      }
      result[row * dimension + column] = value;
      result[column * dimension + row] = value;
    }
  }
  return result;
}

class Matrix {
public:
  explicit Matrix(std::size_t dimension) {
    fmpz_mat_init(&value_, static_cast<slong>(dimension),
                  static_cast<slong>(dimension));
  }

  Matrix(const Matrix &) = delete;
  Matrix &operator=(const Matrix &) = delete;

  ~Matrix() { fmpz_mat_clear(&value_); }

  fmpz_mat_struct *get() { return &value_; }
  const fmpz_mat_struct *get() const { return &value_; }

private:
  fmpz_mat_struct value_{};
};

using MatrixPtr = std::unique_ptr<Matrix>;

MatrixPtr to_matrix(const Rows &rows, std::size_t dimension) {
  auto matrix = std::make_unique<Matrix>(dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column < dimension; ++column) {
      fmpz_set_si(fmpz_mat_entry(matrix->get(), static_cast<slong>(row),
                                 static_cast<slong>(column)),
                  static_cast<slong>(rows[row * dimension + column]));
    }
  }
  return matrix;
}

void reduce(fmpz_mat_struct *matrix, fmpz_mat_struct *transform,
            const fmpz_lll_t context) {
  fmpz_mat_one(transform);
  fmpz_lll(matrix, transform, context);
}

void verify_reduction(const Matrix &source, const Matrix &reduced,
                      const Matrix &transform, std::size_t dimension,
                      const fmpz_lll_t context) {
  if (fmpz_lll_is_reduced(reduced.get(), context, 128) == 0) {
    throw std::runtime_error(
        "FLINT returned a matrix without an LLL certificate");
  }

  Matrix product(dimension);
  Matrix transpose(dimension);
  Matrix expected(dimension);
  fmpz_mat_mul(product.get(), transform.get(), source.get());
  fmpz_mat_transpose(transpose.get(), transform.get());
  fmpz_mat_mul(expected.get(), product.get(), transpose.get());
  if (fmpz_mat_equal(expected.get(), reduced.get()) == 0) {
    throw std::runtime_error(
        "FLINT transform does not reproduce the reduced Gram matrix");
  }
}

double median_ns(std::vector<std::chrono::nanoseconds> timings,
                 std::size_t operations) {
  std::sort(timings.begin(), timings.end());
  return static_cast<double>(timings[timings.size() / 2].count()) /
         static_cast<double>(operations);
}

std::pair<double, std::int64_t> benchmark_lll(std::size_t dimension) {
  std::vector<MatrixPtr> sources;
  sources.reserve(lll_cases);
  std::int64_t checksum = 0;
  std::size_t flat_index = 0;
  for (std::size_t benchmark_case = 0; benchmark_case < lll_cases;
       ++benchmark_case) {
    const auto basis = skew_basis(dimension, benchmark_case);
    for (const auto entry : basis) {
      ++flat_index;
      checksum += static_cast<std::int64_t>(flat_index) * entry;
    }
    sources.push_back(to_matrix(gram(basis, dimension), dimension));
  }

  fmpz_lll_t context;
  fmpz_lll_context_init(context, 0.99, std::nextafter(0.5, 1.0), GRAM, EXACT);

  for (const auto &source : sources) {
    Matrix reduced(dimension);
    Matrix transform(dimension);
    fmpz_mat_set(reduced.get(), source->get());
    reduce(reduced.get(), transform.get(), context);
    verify_reduction(*source, reduced, transform, dimension, context);
    result_sink = fmpz_get_si(fmpz_mat_entry(reduced.get(), 0, 0));
  }

  std::vector<std::chrono::nanoseconds> timings;
  timings.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (const auto &source : sources) {
      Matrix reduced(dimension);
      Matrix transform(dimension);
      fmpz_mat_set(reduced.get(), source->get());
      reduce(reduced.get(), transform.get(), context);
      result_sink = fmpz_get_si(fmpz_mat_entry(reduced.get(), 0, 0));
    }
    timings.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start));
  }
  return {median_ns(std::move(timings), lll_cases), checksum};
}

} // namespace

int main() {
  try {
    std::cout << "library,operation,dimension,median_ns,target_fingerprint,"
                 "point_fingerprint,distance_fingerprint\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const int dimension : dimensions) {
      const auto [lll_ns, basis_checksum] =
          benchmark_lll(static_cast<std::size_t>(dimension));
      std::cout << "flint,lll_gram_with_transform," << dimension << ','
                << lll_ns << ',' << basis_checksum << ",0,0\n";
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  flint_cleanup_master();
  return static_cast<int>(result_sink == std::int64_t{-1});
}
