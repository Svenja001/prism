/* NVGT - NonVisual Gaming Toolkit
 * Copyright (c) 2022-2025 Sam Tupy
 * https://nvgt.dev
 * This software is provided "as-is", without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising from the
 * use of this software. Permission is granted to anyone to use this software
 * for any purpose, including commercial applications, and to alter it and
 * redistribute it freely, subject to the following restrictions:
 * 1. The origin of this software must not be misrepresented; you must not claim
 * that you wrote the original software. If you use this software in a product,
 * an acknowledgment in the product documentation would be appreciated but is
 * not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <numeric>
#include <utility>
#if (defined(__x86_64__) || defined(_M_X64)) &&                                \
    (defined(PRISM_FORCE_MANUAL) || !defined(__ELF__))
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
#if (defined(__aarch64__) || defined(_M_ARM64)) &&                             \
    (defined(PRISM_FORCE_MANUAL) || !defined(__ELF__))
#ifdef __APPLE__
#include <sys/sysctl.h>
#elifdef _WIN32
#include <windows.h>
#elifdef __linux__
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif
#ifndef _MSC_VER
#pragma STDC FP_CONTRACT ON
#endif
#if defined(__GNUC__) && !defined(__clang__)
#define PRISM_REASSOC_ATTR                                                     \
  [[gnu::optimize("-fassociative-math", "-fno-signed-zeros",                   \
                  "-fno-trapping-math")]]
#else
#define PRISM_REASSOC_ATTR
#endif
#ifdef __clang__
#define PRISM_REASSOC_PRAGMA _Pragma("clang fp reassociate(on) contract(fast)")
#define PRISM_AVX10 "avx10.2-256"
#else
#define PRISM_REASSOC_PRAGMA
#define PRISM_AVX10 "avx10.2"
#endif
#if !defined(PRISM_FORCE_MANUAL) && defined(__ELF__) &&                        \
    ((defined(__x86_64__) || defined(_M_X64)) ||                               \
     (defined(__aarch64__) && defined(__GNUC__) && !defined(__clang__)))
#define PRISM_TARGET_CLONES 1
#else
#define PRISM_TARGET_CLONES 0
#endif

// Begin NVGT code
double range_convert(double v, double a0, double a1, double b0, double b1) {
  const double t = (v - a0) / (a1 - a0);
  return std::lerp(b0, b1, t);
}

float range_convert(float old_value, float old_min, float old_max,
                    float new_min, float new_max) {
  return (((old_value - old_min) / (old_max - old_min)) * (new_max - new_min)) +
         new_min;
}

float range_convert_midpoint(float old_value, float old_min, float old_midpoint,
                             float old_max, float new_min, float new_midpoint,
                             float new_max) {
  if (old_value <= old_midpoint)
    return range_convert(old_value, old_min, old_midpoint, new_min,
                         new_midpoint);
  else
    return range_convert(old_value, old_midpoint, old_max, new_midpoint,
                         new_max);
}
// End NVGT code

double exp_range_convert(float t, double out_min, double out_mid,
                         double out_max) {
  const double log_min = std::log(out_min);
  const double log_mid = std::log(out_mid);
  const double log_max = std::log(out_max);
  double log_val;
  if (t <= 0.5)
    log_val = log_min + ((log_mid - log_min) * (t / 0.5));
  else
    log_val = log_mid + ((log_max - log_mid) * ((t - 0.5) / 0.5));
  return std::exp(log_val);
}

float exp_range_convert_inv(double val, double out_min, double out_mid,
                            double out_max) {
  const double log_min = std::log(out_min);
  const double log_mid = std::log(out_mid);
  const double log_max = std::log(out_max);
  double log_val = std::log(std::clamp(val, out_min, out_max));
  if (log_val <= log_mid)
    return static_cast<float>(0.5 * (log_val - log_min) / (log_mid - log_min));
  else
    return static_cast<float>(
        0.5 + ((0.5 * (log_val - log_mid)) / (log_max - log_mid)));
}

struct TrimBounds {
  std::size_t start_frame = 0;
  std::size_t end_frame = 0;
  bool speech_detected = false;
  float noise_floor_db = -160.0F;
  float open_thr_db = -160.0F;
  float close_thr_db = -160.0F;
};

struct TrimWorkspace {
  std::vector<float> db;
  std::vector<float> scratch;
};

[[nodiscard]] static inline std::span<const float>
hann_window(std::size_t fade_frames) {
  static thread_local std::vector<float> cache;
  if (cache.size() != fade_frames) {
    cache.resize(fade_frames);
    const auto denom = static_cast<double>(fade_frames - 1);
    for (std::size_t i = 0; i < fade_frames; ++i) {
      const double t = static_cast<double>(i) / denom;
      cache[i] =
          static_cast<float>(0.5 - (0.5 * std::cos(std::numbers::pi * t)));
    }
  }
  return {cache.data(), cache.size()};
}

static inline std::size_t ms_to_frames(float ms, std::size_t sample_rate) {
  const auto f =
      (static_cast<double>(ms) * static_cast<double>(sample_rate)) / 1000.0;
  return static_cast<std::size_t>(std::max(0.0, std::floor(f + 0.5)));
}

static inline float mean_square_to_db(double mean_square) {
  constexpr double eps = 1e-16;
  return static_cast<float>(10.0 * std::log10(mean_square + eps));
}

namespace {
PRISM_REASSOC_ATTR [[gnu::always_inline]] inline void
fill_frame_db_impl(std::span<const float> interleaved, std::size_t channels,
                   std::size_t frame_len, std::size_t hop,
                   std::size_t total_frames, std::span<float> db) {
  const std::size_t n = db.size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t start_frame = i * hop;
    const std::size_t end_frame =
        std::min(start_frame + frame_len, total_frames);
    if (end_frame <= start_frame) {
      db[i] = -160.0F;
      continue;
    }
    const std::size_t nsamples = (end_frame - start_frame) * channels;
    const float *__restrict p = interleaved.data() + (start_frame * channels);
    float sumsq;
    {
      PRISM_REASSOC_PRAGMA
      std::array<float, 16> s{};
      std::size_t k = 0;
      const std::size_t lim = nsamples & ~static_cast<std::size_t>(15);
      for (; k < lim; k += 16)
        for (int j = 0; j < 16; ++j) {
          const float v = p[k + j];
          s[j] += v * v;
        }
      float acc = 0.0F;
      for (int j = 0; j < 16; ++j)
        acc += s[j];
      for (; k < nsamples; ++k) {
        const float v = p[k];
        acc += v * v;
      }
      sumsq = acc;
    }
    db[i] = mean_square_to_db(static_cast<double>(sumsq) /
                              static_cast<double>(nsamples));
  }
}

using prism_fill_fn = void (*)(std::span<const float>, std::size_t, std::size_t,
                               std::size_t, std::size_t, std::span<float>);

#if PRISM_TARGET_CLONES
#if defined(__x86_64__) || defined(_M_X64)
[[gnu::target_clones("arch=x86-64-v4", "arch=x86-64-v3", "arch=x86-64-v2",
                     "default")]]
#else
[[gnu::target_clones("sve2", "sve", "default")]]
#endif
PRISM_REASSOC_ATTR void fill_frame_db(std::span<const float> in, std::size_t ch,
                                      std::size_t fl, std::size_t hop,
                                      std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
#else
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
inline void cpuidex(std::array<int, 4> &info, int leaf, int sub) {
  __cpuidex(info.data(), leaf, sub);
}
inline unsigned long long xgetbv0() { return _xgetbv(0); }
#else
inline void cpuidex(std::array<int, 4> &info, int leaf, int sub) {
  __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
}
inline unsigned long long xgetbv0() {
  unsigned int lo, hi;
  __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif
PRISM_REASSOC_ATTR __attribute__((target(PRISM_AVX10))) void
fill_v5(std::span<const float> in, std::size_t ch, std::size_t fl,
        std::size_t hop, std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
PRISM_REASSOC_ATTR
__attribute__((target("avx512f,avx512bw,avx512cd,avx512dq,avx512vl"))) void
fill_v4(std::span<const float> in, std::size_t ch, std::size_t fl,
        std::size_t hop, std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
PRISM_REASSOC_ATTR __attribute__((target("avx2,fma"))) void
fill_v3(std::span<const float> in, std::size_t ch, std::size_t fl,
        std::size_t hop, std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
PRISM_REASSOC_ATTR __attribute__((target("sse4.2"))) void
fill_v2(std::span<const float> in, std::size_t ch, std::size_t fl,
        std::size_t hop, std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
PRISM_REASSOC_ATTR void fill_base(std::span<const float> in, std::size_t ch,
                                  std::size_t fl, std::size_t hop,
                                  std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}

prism_fill_fn resolve_fill_frames_db_impl() {
  std::array<int, 4> r;
  cpuidex(r, 0, 0);
  const int maxleaf = r[0];
  cpuidex(r, 1, 0);
  const bool sse42 = (r[2] & (1 << 20)) != 0;
  const bool popcnt = (r[2] & (1 << 23)) != 0;
  const bool osxsave = (r[2] & (1 << 27)) != 0;
  const bool avx = (r[2] & (1 << 28)) != 0;
  const bool fma = (r[2] & (1 << 12)) != 0;
  bool ymm = false;
  bool zmm = false;
  bool opmask = false;
  if (osxsave) {
    const unsigned long long xcr0 = xgetbv0();
    ymm = (xcr0 & 0x6) == 0x6;            // XMM|YMM
    opmask = (xcr0 & 0x20) != 0;          // bit5 opmask (k-regs)
    zmm = ymm && ((xcr0 & 0xE0) == 0xE0); // opmask|ZMM_Hi256|Hi16_ZMM
  }
  bool avx2 = false;
  bool f = false;
  bool bw = false;
  bool cd = false;
  bool dq = false;
  bool vl = false;
  bool avx10_256 = false;
  if (maxleaf >= 7) {
    cpuidex(r, 7, 0);
    avx2 = (r[1] & (1 << 5)) != 0;
    f = (r[1] & (1 << 16)) != 0;
    dq = (r[1] & (1 << 17)) != 0;
    cd = (r[1] & (1 << 28)) != 0;
    bw = (r[1] & (1 << 30)) != 0;
    vl = (r[1] & (1 << 31)) != 0;
    cpuidex(r, 7, 1);
    const bool avx10 = (r[3] & (1 << 19)) != 0;
    if (avx10 && maxleaf >= 0x24) {
      cpuidex(r, 0x24, 0);
      const int ver = r[1] & 0xFF;
      const bool vl256 = (r[1] & (1 << 17)) != 0;
      avx10_256 = (ver >= 2) && vl256 && ymm && opmask;
    }
  }
  if (avx10_256)
    return &fill_v5;
  if (f && bw && cd && dq && vl && zmm)
    return &fill_v4;
  if (avx && avx2 && fma && ymm)
    return &fill_v3;
  if (sse42 && popcnt)
    return &fill_v2;
  return &fill_base;
}
#elif defined(__aarch64__) || defined(_M_ARM64)
PRISM_REASSOC_ATTR __attribute__((target("arch=armv8-a+sve2"))) void
fill_sve2(std::span<const float> in, std::size_t ch, std::size_t fl,
          std::size_t hop, std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
PRISM_REASSOC_ATTR __attribute__((target("arch=armv8-a+sve"))) void
fill_sve(std::span<const float> in, std::size_t ch, std::size_t fl,
         std::size_t hop, std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
PRISM_REASSOC_ATTR void fill_neon(std::span<const float> in, std::size_t ch,
                                  std::size_t fl, std::size_t hop,
                                  std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
prism_fill_fn resolve_fill_frames_db_impl() {
  bool sve = false;
  bool sve2 = false;
#ifdef __APPLE__
  auto has = [](const char *name) {
    int v = 0;
    std::size_t sz = sizeof(v);
    return sysctlbyname(name, &v, &sz, nullptr, 0) == 0 && v != 0;
  };
  sve = has("hw.optional.arm.FEAT_SVE");
  sve2 = has("hw.optional.arm.FEAT_SVE2");
#elifdef _WIN32
#ifdef PF_ARM_SVE_INSTRUCTIONS_AVAILABLE
  sve = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#ifdef PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE
  sve2 = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#elifdef __linux__
  const unsigned long hw = getauxval(AT_HWCAP);
  const unsigned long hw2 = getauxval(AT_HWCAP2);
#ifdef HWCAP_SVE
  sve = (hw & HWCAP_SVE) != 0;
#endif
#ifdef HWCAP2_SVE2
  sve2 = (hw2 & HWCAP2_SVE2) != 0;
#endif
#endif
  if (sve2)
    return &fill_sve2;
  if (sve)
    return &fill_sve;
  return &fill_neon;
}
#else
PRISM_REASSOC_ATTR void fill_base(std::span<const float> in, std::size_t ch,
                                  std::size_t fl, std::size_t hop,
                                  std::size_t tot, std::span<float> db) {
  fill_frame_db_impl(in, ch, fl, hop, tot, db);
}
prism_fill_fn resolve_fill_frames_db_impl() { return &fill_base; }
#endif
}

static void fill_frame_db(std::span<const float> in, std::size_t ch,
                          std::size_t fl, std::size_t hop, std::size_t tot,
                          std::span<float> db) {
  static const prism_fill_fn impl = resolve_fill_frames_db_impl();
  impl(in, ch, fl, hop, tot, db);
}
#endif
#undef PRISM_TARGET_CLONES
#undef PRISM_REASSOC_ATTR
#undef PRISM_REASSOC_PRAGMA

static inline float percentile(std::span<const float> x, float p,
                               std::vector<float> &scratch) {
  if (x.empty())
    return -160.0F;
  p = std::clamp(p, 0.0F, 1.0F);
  scratch.assign(x.begin(), x.end());
  const std::size_t n = scratch.size();
  if (n == 1)
    return scratch[0];
  const auto k = static_cast<std::size_t>(
      std::floor(static_cast<double>(p) * static_cast<double>(n - 1)));
  std::nth_element(scratch.begin(),
                   scratch.begin() + static_cast<std::ptrdiff_t>(k),
                   scratch.end());
  return scratch[k];
}

static inline void apply_fade_in(std::span<float> interleaved,
                                 std::size_t channels,
                                 std::size_t fade_frames) {
  if (fade_frames == 0 || channels == 0)
    return;
  const auto total_frames = interleaved.size() / channels;
  fade_frames = std::min(fade_frames, total_frames);
  if (fade_frames <= 1)
    return;
  const auto window = hann_window(fade_frames);
  const float *__restrict w = window.data();
  if (channels == 1) {
    float *__restrict p = interleaved.data();
    for (std::size_t i = 0; i < fade_frames; ++i)
      p[i] *= w[i];
    return;
  }
  if (channels == 2) {
    float *__restrict p = interleaved.data();
    for (std::size_t i = 0; i < fade_frames; ++i) {
      const float g = w[i];
      p[(2 * i) + 0] *= g;
      p[(2 * i) + 1] *= g;
    }
    return;
  }
  for (std::size_t i = 0; i < fade_frames; ++i) {
    const float g = w[i];
    const auto base = i * channels;
    for (std::size_t ch = 0; ch < channels; ++ch)
      interleaved[base + ch] *= g;
  }
}

static inline void apply_fade_out(std::span<float> interleaved,
                                  std::size_t channels,
                                  std::size_t fade_frames) {
  if (fade_frames == 0 || channels == 0)
    return;
  const auto total_frames = interleaved.size() / channels;
  fade_frames = std::min(fade_frames, total_frames);
  if (fade_frames == 0)
    return;
  const auto start = total_frames - fade_frames;
  if (fade_frames == 1) {
    const auto base = start * channels;
    for (std::size_t ch = 0; ch < channels; ++ch)
      interleaved[base + ch] = 0.0F;
    return;
  }
  const auto window = hann_window(fade_frames);
  const float *__restrict w = window.data();
  const std::size_t last = fade_frames - 1;
  if (channels == 1) {
    float *__restrict p = interleaved.data() + start;
    for (std::size_t i = 0; i < fade_frames; ++i)
      p[i] *= w[last - i];
    return;
  }
  if (channels == 2) {
    float *__restrict p = interleaved.data() + (start * 2);
    for (std::size_t i = 0; i < fade_frames; ++i) {
      const float g = w[last - i];
      p[(2 * i) + 0] *= g;
      p[(2 * i) + 1] *= g;
    }
    return;
  }
  for (std::size_t i = 0; i < fade_frames; ++i) {
    const float g = w[last - i];
    const auto base = (start + i) * channels;
    for (std::size_t ch = 0; ch < channels; ++ch)
      interleaved[base + ch] *= g;
  }
}

static inline double frame_abs_sum(std::span<const float> interleaved,
                                   std::size_t frame, std::size_t channels) {
  const auto base = frame * channels;
  if (channels == 1) {
    return std::abs(interleaved[base]);
  }
  if (channels == 2) {
    return static_cast<double>(std::abs(interleaved[base])) +
           static_cast<double>(std::abs(interleaved[base + 1]));
  }
  double s = 0.0;
  for (std::size_t ch = 0; ch < channels; ++ch)
    s += std::abs(interleaved[base + ch]);
  return s;
}

static inline std::size_t snap_start(std::span<const float> interleaved,
                                     std::size_t target,
                                     std::size_t total_frames,
                                     std::size_t channels, std::size_t search) {
  if (search == 0 || total_frames == 0)
    return std::min(target, total_frames);
  target = std::min(target, total_frames);
  const auto begin = (target > search) ? (target - search) : 0;
  const auto end = std::min(total_frames, target + search + 1);
  auto best = target;
  auto best_score = std::numeric_limits<double>::infinity();
  for (std::size_t f = begin; f < end; ++f) {
    const auto s = frame_abs_sum(interleaved, f, channels);
    if (s < best_score) {
      best_score = s;
      best = f;
    }
  }
  return best;
}

static inline std::size_t snap_end(std::span<const float> interleaved,
                                   std::size_t target_excl,
                                   std::size_t total_frames,
                                   std::size_t channels, std::size_t search) {
  if (search == 0 || total_frames == 0)
    return std::min(target_excl, total_frames);
  target_excl = std::min(target_excl, total_frames);
  const auto begin = (target_excl > search) ? (target_excl - search) : 0;
  const auto end = std::min(total_frames, target_excl + search);
  auto best = target_excl;
  auto best_score = std::numeric_limits<double>::infinity();
  for (std::size_t b = begin; b <= end; ++b) {
    double s = 0.0;
    if (b > 0)
      s += frame_abs_sum(interleaved, b - 1, channels);
    if (b < total_frames)
      s += frame_abs_sum(interleaved, b, channels);
    if (s < best_score) {
      best_score = s;
      best = b;
    }
  }
  return best;
}

static inline TrimBounds
compute_trim_bounds_rms_gate(std::span<const float> samples_interleaved,
                             std::size_t channels, std::size_t sample_rate,
                             const TrimParams &P = {}) {
  TrimBounds R{};
  if (channels == 0 || sample_rate == 0)
    return R;
  if (samples_interleaved.empty())
    return R;
  if (samples_interleaved.size() % channels != 0)
    return R;
  const auto total_frames = samples_interleaved.size() / channels;
  const auto frame_len =
      std::max<std::size_t>(1, ms_to_frames(P.frame_ms, sample_rate));
  const auto hop =
      std::max<std::size_t>(1, ms_to_frames(P.hop_ms, sample_rate));
  const auto n = (total_frames <= frame_len)
                     ? std::size_t{1}
                     : (std::size_t{1} + ((total_frames - frame_len) / hop));
  static thread_local TrimWorkspace W;
  W.db.resize(n);
  auto &db = W.db;
  fill_frame_db(samples_interleaved, channels, frame_len, hop, total_frames,
                std::span<float>{db});
  const auto head_frames = std::min<std::size_t>(
      n, std::max<std::size_t>(std::size_t{1},
                               ms_to_frames(P.head_ms, sample_rate) / hop));
  const auto tail_frames = std::min<std::size_t>(
      n, std::max<std::size_t>(std::size_t{1},
                               ms_to_frames(P.tail_ms, sample_rate) / hop));
  const std::span<const float> head(db.data(), head_frames);
  const std::span<const float> tail(db.data() + (n - tail_frames), tail_frames);
  W.scratch.reserve(std::max(head_frames, tail_frames));
  float floor_db = std::min(percentile(head, 0.20F, W.scratch),
                            percentile(tail, 0.20F, W.scratch));
  floor_db = std::clamp(floor_db, P.min_floor_db, P.max_floor_db);
  const float open_thr = floor_db + P.open_db;
  const float close_thr = floor_db + P.close_db;
  R.noise_floor_db = floor_db;
  R.open_thr_db = open_thr;
  R.close_thr_db = close_thr;
  const auto min_on = std::max(1, P.min_speech_frames);
  const auto min_off = std::max(1, P.min_silence_frames);
  bool in_speech = false;
  int on_run = 0;
  int off_run = 0;
  std::size_t start_idx = 0;
  std::size_t end_excl_idx = n;
  bool have_start = false;
  for (std::size_t i = 0; i < n; ++i) {
    const float v = db[i];
    if (!in_speech) {
      if (v >= open_thr) {
        if (++on_run >= min_on) {
          in_speech = true;
          off_run = 0;
          const auto onset = i + 1 - static_cast<std::size_t>(min_on);
          if (!have_start) {
            start_idx = onset;
            have_start = true;
          }
          on_run = 0;
        }
      } else {
        on_run = 0;
      }
    } else {
      if (v <= close_thr) {
        if (++off_run >= min_off) {
          in_speech = false;
          on_run = 0;
          const auto silence_start = i + 1 - static_cast<std::size_t>(min_off);
          end_excl_idx = std::min(end_excl_idx, silence_start);
          off_run = 0;
        }
      } else {
        off_run = 0;
        end_excl_idx = n;
      }
    }
  }
  if (!have_start) {
    R.speech_detected = false;
    R.start_frame = 0;
    R.end_frame = total_frames;
    return R;
  }
  R.speech_detected = true;
  auto start_frame = start_idx * hop;
  auto end_frame_excl =
      (end_excl_idx >= n) ? total_frames : (end_excl_idx * hop);
  const auto preroll = ms_to_frames(P.preroll_ms, sample_rate);
  const auto postroll = ms_to_frames(P.postroll_ms, sample_rate);
  start_frame = (start_frame > preroll) ? (start_frame - preroll) : 0;
  end_frame_excl = std::min(total_frames, end_frame_excl + postroll);
  const auto search = ms_to_frames(P.boundary_search_ms, sample_rate);
  start_frame = snap_start(samples_interleaved, start_frame, total_frames,
                           channels, search);
  end_frame_excl = snap_end(samples_interleaved, end_frame_excl, total_frames,
                            channels, search);
  start_frame = std::min(start_frame, total_frames);
  end_frame_excl = std::min(end_frame_excl, total_frames);
  if (end_frame_excl <= start_frame) {
    R.start_frame = 0;
    R.end_frame = 0;
    return R;
  }
  R.start_frame = start_frame;
  R.end_frame = end_frame_excl;
  return R;
}
std::vector<float>
trim_silence_rms_gate(std::span<const float> samples_interleaved,
                      std::size_t channels, std::size_t sample_rate,
                      const TrimParams &P) {
  if (channels == 0 || sample_rate == 0)
    return {samples_interleaved.begin(), samples_interleaved.end()};
  if (samples_interleaved.empty() ||
      (samples_interleaved.size() % channels) != 0)
    return {samples_interleaved.begin(), samples_interleaved.end()};
  const auto bounds = compute_trim_bounds_rms_gate(samples_interleaved,
                                                   channels, sample_rate, P);
  if (!bounds.speech_detected)
    return {samples_interleaved.begin(), samples_interleaved.end()};
  const auto start = bounds.start_frame;
  const auto end = bounds.end_frame;
  std::vector<float> out;
  out.resize((end - start) * channels);
  std::copy(samples_interleaved.begin() +
                static_cast<std::ptrdiff_t>(start * channels),
            samples_interleaved.begin() +
                static_cast<std::ptrdiff_t>(end * channels),
            out.begin());
  const auto fade_frames = ms_to_frames(P.fade_ms, sample_rate);
  std::span<float> out_span(out);
  apply_fade_in(out_span, channels, fade_frames);
  apply_fade_out(out_span, channels, fade_frames);
  return out;
}

TrimView trim_silence_rms_gate_inplace(std::span<float> interleaved,
                                       std::size_t channels,
                                       std::size_t sample_rate,
                                       const TrimParams &P) {
  TrimView r{.view = interleaved, .speech_detected = false};
  if (channels == 0 || sample_rate == 0)
    return r;
  if (interleaved.empty() || (interleaved.size() % channels) != 0)
    return r;
  const auto bounds = compute_trim_bounds_rms_gate(
      std::span<const float>(interleaved.data(), interleaved.size()), channels,
      sample_rate, P);
  if (!bounds.speech_detected)
    return r;
  const std::size_t start = bounds.start_frame * channels;
  const std::size_t end = bounds.end_frame * channels;
  if (end <= start || end > interleaved.size())
    return r;
  r.speech_detected = true;
  r.view = interleaved.subspan(start, end - start);
  const auto fade_frames = ms_to_frames(P.fade_ms, sample_rate);
  apply_fade_in(r.view, channels, fade_frames);
  apply_fade_out(r.view, channels, fade_frames);
  return r;
}