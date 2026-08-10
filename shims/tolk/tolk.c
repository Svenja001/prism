// SPDX-License-Identifier: MPL-2.0

#include "tolk.h"
#include "lock.h"
#include "thread_safety.h"
#include <prism.h>
#include <simdutf_c.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <uchar.h>
#include <wchar.h>
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif
#ifdef _WIN32
static const PrismBackendId default_tts_backend = PRISM_BACKEND_SAPI;
#elif defined(__APPLE__)
static const PrismBackendId default_tts_backend = PRISM_BACKEND_AV_SPEECH;
#elif defined(__ANDROID__)
static const PrismBackendId default_tts_backend = PRISM_BACKEND_ANDROID_TTS;
#elif defined(__EMSCRIPTEN__)
static const PrismBackendId default_tts_backend = PRISM_BACKEND_WEB_SPEECH;
#else
static const PrismBackendId default_tts_backend =
    PRISM_BACKEND_SPEECH_DISPATCHER;
#endif

/*
 * The following is extremely nasty, but MSVC, for some reason, does not
 * support nullptr in C23.
 */
#ifdef _MSC_VER
#define NULL_CONSTANT NULL
#else
#define NULL_CONSTANT nullptr
#endif

static fast_lock lock = FAST_LOCK_INIT;
static PrismContext *ctx TSA_GUARDED_BY(lock);
static PrismBackend *backend TSA_GUARDED_BY(lock);
static PrismBackend *sapi_backend TSA_GUARDED_BY(lock);
static wchar_t *backend_name TSA_GUARDED_BY(lock);
static wchar_t *sapi_backend_name TSA_GUARDED_BY(lock);
static bool loaded TSA_GUARDED_BY(lock);
static bool prefer_sapi TSA_GUARDED_BY(lock);

static inline char *wchar_to_utf8(const wchar_t *src) {
  if (src == NULL_CONSTANT)
    return NULL_CONSTANT;
#if WCHAR_MAX <= 0xFFFFu
  const char16_t *in = (const char16_t *)src;
  size_t in_len = 0;
  while (in[in_len] != 0)
    ++in_len;
  const size_t out_len = simdutf_utf8_length_from_utf16(in, in_len);
  char *buf = malloc(out_len + 1);
  if (buf == NULL_CONSTANT)
    return NULL_CONSTANT;
  const size_t written = simdutf_convert_utf16_to_utf8(in, in_len, buf);
  if (written == 0 && in_len != 0) {
    free(buf);
    return NULL_CONSTANT;
  }
  buf[written] = '\0';
  return buf;
#else
  const char32_t *in = (const char32_t *)src;
  size_t in_len = 0;
  while (in[in_len] != 0)
    ++in_len;
  const size_t out_len = simdutf_utf8_length_from_utf32(in, in_len);
  char *buf = malloc(out_len + 1);
  if (buf == NULL_CONSTANT)
    return NULL_CONSTANT;
  const size_t written = simdutf_convert_utf32_to_utf8(in, in_len, buf);
  if (written == 0 && in_len != 0) {
    free(buf);
    return NULL_CONSTANT;
  }
  buf[written] = '\0';
  return buf;
#endif
}

static inline wchar_t *utf8_to_wchar(const char *src) {
  if (src == NULL_CONSTANT)
    return NULL_CONSTANT;
  const size_t in_len = strlen(src);
#if WCHAR_MAX <= 0xFFFFu
  const size_t out_len = simdutf_utf16_length_from_utf8(src, in_len);
  wchar_t *buf = malloc((out_len + 1) * sizeof(wchar_t));
  if (buf == NULL_CONSTANT)
    return NULL_CONSTANT;
  const size_t written =
      simdutf_convert_utf8_to_utf16(src, in_len, (char16_t *)buf);
  if (written == 0 && in_len != 0) {
    free(buf);
    return NULL_CONSTANT;
  }
  buf[written] = L'\0';
  return buf;
#else
  const size_t out_len = simdutf_utf32_length_from_utf8(src, in_len);
  wchar_t *buf = malloc((out_len + 1) * sizeof(wchar_t));
  if (buf == NULL_CONSTANT)
    return NULL_CONSTANT;
  const size_t written =
      simdutf_convert_utf8_to_utf32(src, in_len, (char32_t *)buf);
  if (written == 0 && in_len != 0) {
    free(buf);
    return NULL_CONSTANT;
  }
  buf[written] = L'\0';
  return buf;
#endif
}

TOLK_API void TOLK_CALL Tolk_Load(void) {
  fast_lock_acquire(&lock);
  if (loaded) {
    fast_lock_release(&lock);
    return;
  }
  PrismConfig cfg = prism_config_init();
  ctx = prism_init(&cfg);
  if (ctx == NULL_CONSTANT) {
    fast_lock_release(&lock);
    return;
  }
  backend = prism_registry_create_best(ctx);
  if (backend != NULL_CONSTANT) {
    const PrismError res = prism_backend_initialize(backend);
    if (res != PRISM_OK && res != PRISM_ERROR_ALREADY_INITIALIZED) {
      prism_backend_free(backend);
      backend = NULL_CONSTANT;
    }
  }
  sapi_backend = prism_registry_create(ctx, default_tts_backend);
  if (sapi_backend != NULL_CONSTANT) {
    const PrismError res = prism_backend_initialize(sapi_backend);
    if (res != PRISM_OK && res != PRISM_ERROR_ALREADY_INITIALIZED) {
      prism_backend_free(sapi_backend);
      sapi_backend = NULL_CONSTANT;
    }
  }
  if (backend != NULL_CONSTANT) {
    backend_name = utf8_to_wchar(prism_backend_name(backend));
  }
  if (sapi_backend != NULL_CONSTANT) {
    sapi_backend_name = utf8_to_wchar(prism_backend_name(sapi_backend));
  }
  if (backend != NULL_CONSTANT || sapi_backend != NULL_CONSTANT) {
    loaded = true;
  } else {
    prism_shutdown(ctx);
    ctx = NULL_CONSTANT;
  }
  fast_lock_release(&lock);
}

TOLK_API bool TOLK_CALL Tolk_IsLoaded(void) {
  fast_lock_acquire(&lock);
  const bool result = loaded;
  fast_lock_release(&lock);
  return result;
}

TOLK_API void TOLK_CALL Tolk_Unload(void) {
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    return;
  }
  if (backend != NULL_CONSTANT) {
    prism_backend_free(backend);
    backend = NULL_CONSTANT;
  }
  if (sapi_backend != NULL_CONSTANT) {
    prism_backend_free(sapi_backend);
    sapi_backend = NULL_CONSTANT;
  }
  if (ctx != NULL_CONSTANT) {
    prism_shutdown(ctx);
    ctx = NULL_CONSTANT;
  }
  free(backend_name);
  backend_name = NULL_CONSTANT;
  free(sapi_backend_name);
  sapi_backend_name = NULL_CONSTANT;
  loaded = false;
  fast_lock_release(&lock);
}

TOLK_API void TOLK_CALL Tolk_TrySAPI(bool trySAPI) { (void)trySAPI; }

TOLK_API void TOLK_CALL Tolk_PreferSAPI(bool preferSAPI) {
  fast_lock_acquire(&lock);
  prefer_sapi = preferSAPI;
  fast_lock_release(&lock);
}

TOLK_API const wchar_t *TOLK_CALL Tolk_DetectScreenReader(void) {
  static _Thread_local wchar_t buf[256];
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    return NULL_CONSTANT;
  }
  const wchar_t *name = prefer_sapi ? sapi_backend_name : backend_name;
  if (name == NULL_CONSTANT) {
    fast_lock_release(&lock);
    return NULL_CONSTANT;
  }
  wcsncpy(buf, name, 255);
  buf[255] = L'\0';
  fast_lock_release(&lock);
  return buf;
}

TOLK_API bool TOLK_CALL Tolk_HasSpeech(void) {
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    return false;
  }
  PrismBackend *b = prefer_sapi ? sapi_backend : backend;
  if (b == NULL_CONSTANT) {
    fast_lock_release(&lock);
    return false;
  }
  const uint64_t features = prism_backend_get_features(b);
  fast_lock_release(&lock);
  return (features & PRISM_BACKEND_SUPPORTS_SPEAK) != 0;
}

TOLK_API bool TOLK_CALL Tolk_HasBraille(void) {
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    return false;
  }
  PrismBackend *b = prefer_sapi ? sapi_backend : backend;
  if (b == NULL_CONSTANT) {
    fast_lock_release(&lock);
    return false;
  }
  const uint64_t features = prism_backend_get_features(b);
  fast_lock_release(&lock);
  return (features & PRISM_BACKEND_SUPPORTS_BRAILLE) != 0;
}

TOLK_API bool TOLK_CALL Tolk_Output(const wchar_t *str, bool interrupt) {
  if (str == NULL_CONSTANT)
    return false;
  char *utf8 = wchar_to_utf8(str);
  if (utf8 == NULL_CONSTANT)
    return false;
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    free(utf8);
    return false;
  }
  PrismBackend *b = prefer_sapi ? sapi_backend : backend;
  PrismError err = PRISM_ERROR_NOT_INITIALIZED;
  if (b != NULL_CONSTANT)
    err = prism_backend_output(b, utf8, interrupt);
  fast_lock_release(&lock);
  free(utf8);
  return err == PRISM_OK;
}

TOLK_API bool TOLK_CALL Tolk_Speak(const wchar_t *str, bool interrupt) {
  if (str == NULL_CONSTANT)
    return false;
  char *utf8 = wchar_to_utf8(str);
  if (utf8 == NULL_CONSTANT)
    return false;
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    free(utf8);
    return false;
  }
  PrismBackend *b = prefer_sapi ? sapi_backend : backend;
  PrismError err = PRISM_ERROR_NOT_INITIALIZED;
  if (b != NULL_CONSTANT)
    err = prism_backend_speak(b, utf8, interrupt);
  fast_lock_release(&lock);
  free(utf8);
  return err == PRISM_OK;
}

TOLK_API bool TOLK_CALL Tolk_Braille(const wchar_t *str) {
  if (str == NULL_CONSTANT)
    return false;
  char *utf8 = wchar_to_utf8(str);
  if (utf8 == NULL_CONSTANT)
    return false;
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    free(utf8);
    return false;
  }
  PrismBackend *b = prefer_sapi ? sapi_backend : backend;
  PrismError err = PRISM_ERROR_NOT_INITIALIZED;
  if (b != NULL_CONSTANT)
    err = prism_backend_braille(b, utf8);
  fast_lock_release(&lock);
  free(utf8);
  return err == PRISM_OK;
}

TOLK_API bool TOLK_CALL Tolk_IsSpeaking(void) {
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    return false;
  }
  PrismBackend *b = prefer_sapi ? sapi_backend : backend;
  if (b == NULL_CONSTANT) {
    fast_lock_release(&lock);
    return false;
  }
  bool speaking = false;
  const PrismError err = prism_backend_is_speaking(b, &speaking);
  fast_lock_release(&lock);
  if (err != PRISM_OK)
    return false;
  return speaking;
}

TOLK_API bool TOLK_CALL Tolk_Silence(void) {
  fast_lock_acquire(&lock);
  if (!loaded) {
    fast_lock_release(&lock);
    return false;
  }
  PrismBackend *b = prefer_sapi ? sapi_backend : backend;
  PrismError err = PRISM_ERROR_NOT_INITIALIZED;
  if (b != NULL_CONSTANT)
    err = prism_backend_stop(b);
  fast_lock_release(&lock);
  return err == PRISM_OK;
}
