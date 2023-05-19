#define PY_SSIZE_T_CLEAN
#include "Python.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#elif PY_VERSION_HEX < 0x02060000 || (0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03030000)
    #error Cython requires Python 2.6+ or Python 3.3+.
#else
#define CYTHON_ABI "0_28_5"
#define CYTHON_FUTURE_DIVISION 0
#include <stddef.h>
#ifndef offsetof
  #define offsetof(type, member) ( (size_t) & ((type*)0) -> member )
#endif
#if !defined(WIN32) && !defined(MS_WINDOWS)
  #ifndef __stdcall
    #define __stdcall
  #endif
  #ifndef __cdecl
    #define __cdecl
  #endif
  #ifndef __fastcall
    #define __fastcall
  #endif
#endif
#ifndef DL_IMPORT
  #define DL_IMPORT(t) t
#endif
#ifndef DL_EXPORT
  #define DL_EXPORT(t) t
#endif
#define __PYX_COMMA ,
#ifndef HAVE_LONG_LONG
  #if PY_VERSION_HEX >= 0x02070000
    #define HAVE_LONG_LONG
  #endif
#endif
#ifndef PY_LONG_LONG
  #define PY_LONG_LONG LONG_LONG
#endif
#ifndef Py_HUGE_VAL
  #define Py_HUGE_VAL HUGE_VAL
#endif
#ifdef PYPY_VERSION
  #define CYTHON_COMPILING_IN_PYPY 1
  #define CYTHON_COMPILING_IN_PYSTON 0
  #define CYTHON_COMPILING_IN_CPYTHON 0
  #undef CYTHON_USE_TYPE_SLOTS
  #define CYTHON_USE_TYPE_SLOTS 0
  #undef CYTHON_USE_PYTYPE_LOOKUP
  #define CYTHON_USE_PYTYPE_LOOKUP 0
  #if PY_VERSION_HEX < 0x03050000
    #undef CYTHON_USE_ASYNC_SLOTS
    #define CYTHON_USE_ASYNC_SLOTS 0
  #elif !defined(CYTHON_USE_ASYNC_SLOTS)
    #define CYTHON_USE_ASYNC_SLOTS 1
  #endif
  #undef CYTHON_USE_PYLIST_INTERNALS
  #define CYTHON_USE_PYLIST_INTERNALS 0
  #undef CYTHON_USE_UNICODE_INTERNALS
  #define CYTHON_USE_UNICODE_INTERNALS 0
  #undef CYTHON_USE_UNICODE_WRITER
  #define CYTHON_USE_UNICODE_WRITER 0
  #undef CYTHON_USE_PYLONG_INTERNALS
  #define CYTHON_USE_PYLONG_INTERNALS 0
  #undef CYTHON_AVOID_BORROWED_REFS
  #define CYTHON_AVOID_BORROWED_REFS 1
  #undef CYTHON_ASSUME_SAFE_MACROS
  #define CYTHON_ASSUME_SAFE_MACROS 0
  #undef CYTHON_UNPACK_METHODS
  #define CYTHON_UNPACK_METHODS 0
  #undef CYTHON_FAST_THREAD_STATE
  #define CYTHON_FAST_THREAD_STATE 0
  #undef CYTHON_FAST_PYCALL
  #define CYTHON_FAST_PYCALL 0
  #undef CYTHON_PEP489_MULTI_PHASE_INIT
  #define CYTHON_PEP489_MULTI_PHASE_INIT 0
  #undef CYTHON_USE_TP_FINALIZE
  #define CYTHON_USE_TP_FINALIZE 0
#elif defined(PYSTON_VERSION)
  #define CYTHON_COMPILING_IN_PYPY 0
  #define CYTHON_COMPILING_IN_PYSTON 1
  #define CYTHON_COMPILING_IN_CPYTHON 0
  #ifndef CYTHON_USE_TYPE_SLOTS
    #define CYTHON_USE_TYPE_SLOTS 1
  #endif
  #undef CYTHON_USE_PYTYPE_LOOKUP
  #define CYTHON_USE_PYTYPE_LOOKUP 0
  #undef CYTHON_USE_ASYNC_SLOTS
  #define CYTHON_USE_ASYNC_SLOTS 0
  #undef CYTHON_USE_PYLIST_INTERNALS
  #define CYTHON_USE_PYLIST_INTERNALS 0
  #ifndef CYTHON_USE_UNICODE_INTERNALS
    #define CYTHON_USE_UNICODE_INTERNALS 1
  #endif
  #undef CYTHON_USE_UNICODE_WRITER
  #define CYTHON_USE_UNICODE_WRITER 0
  #undef CYTHON_USE_PYLONG_INTERNALS
  #define CYTHON_USE_PYLONG_INTERNALS 0
  #ifndef CYTHON_AVOID_BORROWED_REFS
    #define CYTHON_AVOID_BORROWED_REFS 0
  #endif
  #ifndef CYTHON_ASSUME_SAFE_MACROS
    #define CYTHON_ASSUME_SAFE_MACROS 1
  #endif
  #ifndef CYTHON_UNPACK_METHODS
    #define CYTHON_UNPACK_METHODS 1
  #endif
  #undef CYTHON_FAST_THREAD_STATE
  #define CYTHON_FAST_THREAD_STATE 0
  #undef CYTHON_FAST_PYCALL
  #define CYTHON_FAST_PYCALL 0
  #undef CYTHON_PEP489_MULTI_PHASE_INIT
  #define CYTHON_PEP489_MULTI_PHASE_INIT 0
  #undef CYTHON_USE_TP_FINALIZE
  #define CYTHON_USE_TP_FINALIZE 0
#else
  #define CYTHON_COMPILING_IN_PYPY 0
  #define CYTHON_COMPILING_IN_PYSTON 0
  #define CYTHON_COMPILING_IN_CPYTHON 1
  #ifndef CYTHON_USE_TYPE_SLOTS
    #define CYTHON_USE_TYPE_SLOTS 1
  #endif
  #if PY_VERSION_HEX < 0x02070000
    #undef CYTHON_USE_PYTYPE_LOOKUP
    #define CYTHON_USE_PYTYPE_LOOKUP 0
  #elif !defined(CYTHON_USE_PYTYPE_LOOKUP)
    #define CYTHON_USE_PYTYPE_LOOKUP 1
  #endif
  #if PY_MAJOR_VERSION < 3
    #undef CYTHON_USE_ASYNC_SLOTS
    #define CYTHON_USE_ASYNC_SLOTS 0
  #elif !defined(CYTHON_USE_ASYNC_SLOTS)
    #define CYTHON_USE_ASYNC_SLOTS 1
  #endif
  #if PY_VERSION_HEX < 0x02070000
    #undef CYTHON_USE_PYLONG_INTERNALS
    #define CYTHON_USE_PYLONG_INTERNALS 0
  #elif !defined(CYTHON_USE_PYLONG_INTERNALS)
    #define CYTHON_USE_PYLONG_INTERNALS 1
  #endif
  #ifndef CYTHON_USE_PYLIST_INTERNALS
    #define CYTHON_USE_PYLIST_INTERNALS 1
  #endif
  #ifndef CYTHON_USE_UNICODE_INTERNALS
    #define CYTHON_USE_UNICODE_INTERNALS 1
  #endif
  #if PY_VERSION_HEX < 0x030300F0
    #undef CYTHON_USE_UNICODE_WRITER
    #define CYTHON_USE_UNICODE_WRITER 0
  #elif !defined(CYTHON_USE_UNICODE_WRITER)
    #define CYTHON_USE_UNICODE_WRITER 1
  #endif
  #ifndef CYTHON_AVOID_BORROWED_REFS
    #define CYTHON_AVOID_BORROWED_REFS 0
  #endif
  #ifndef CYTHON_ASSUME_SAFE_MACROS
    #define CYTHON_ASSUME_SAFE_MACROS 1
  #endif
  #ifndef CYTHON_UNPACK_METHODS
    #define CYTHON_UNPACK_METHODS 1
  #endif
  #ifndef CYTHON_FAST_THREAD_STATE
    #define CYTHON_FAST_THREAD_STATE 1
  #endif
  #ifndef CYTHON_FAST_PYCALL
    #define CYTHON_FAST_PYCALL 1
  #endif
  #ifndef CYTHON_PEP489_MULTI_PHASE_INIT
    #define CYTHON_PEP489_MULTI_PHASE_INIT (0 && PY_VERSION_HEX >= 0x03050000)
  #endif
  #ifndef CYTHON_USE_TP_FINALIZE
    #define CYTHON_USE_TP_FINALIZE (PY_VERSION_HEX >= 0x030400a1)
  #endif
#endif
#if !defined(CYTHON_FAST_PYCCALL)
#define CYTHON_FAST_PYCCALL  (CYTHON_FAST_PYCALL && PY_VERSION_HEX >= 0x030600B1)
#endif
#if CYTHON_USE_PYLONG_INTERNALS
  #include "longintrepr.h"
  #undef SHIFT
  #undef BASE
  #undef MASK
#endif
#ifndef __has_attribute
  #define __has_attribute(x) 0
#endif
#ifndef __has_cpp_attribute
  #define __has_cpp_attribute(x) 0
#endif
#ifndef CYTHON_RESTRICT
  #if defined(__GNUC__)
    #define CYTHON_RESTRICT __restrict__
  #elif defined(_MSC_VER) && _MSC_VER >= 1400
    #define CYTHON_RESTRICT __restrict
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_RESTRICT restrict
  #else
    #define CYTHON_RESTRICT
  #endif
#endif
#ifndef CYTHON_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define CYTHON_UNUSED __attribute__ ((__unused__))
#   else
#     define CYTHON_UNUSED
#   endif
# elif defined(__ICC) || (defined(__INTEL_COMPILER) && !defined(_MSC_VER))
#   define CYTHON_UNUSED __attribute__ ((__unused__))
# else
#   define CYTHON_UNUSED
# endif
#endif
#ifndef CYTHON_MAYBE_UNUSED_VAR
#  if defined(__cplusplus)
     template<class T> void CYTHON_MAYBE_UNUSED_VAR( const T& ) { }
#  else
#    define CYTHON_MAYBE_UNUSED_VAR(x) (void)(x)
#  endif
#endif
#ifndef CYTHON_NCP_UNUSED
# if CYTHON_COMPILING_IN_CPYTHON
#  define CYTHON_NCP_UNUSED
# else
#  define CYTHON_NCP_UNUSED CYTHON_UNUSED
# endif
#endif
#define __Pyx_void_to_None(void_result) ((void)(void_result), Py_INCREF(Py_None), Py_None)
#ifdef _MSC_VER
    #ifndef _MSC_STDINT_H_
        #if _MSC_VER < 1300
           typedef unsigned char     uint8_t;
           typedef unsigned int      uint32_t;
        #else
           typedef unsigned __int8   uint8_t;
           typedef unsigned __int32  uint32_t;
        #endif
    #endif
#else
   #include <stdint.h>
#endif
#ifndef CYTHON_FALLTHROUGH
  #if defined(__cplusplus) && __cplusplus >= 201103L
    #if __has_cpp_attribute(fallthrough)
      #define CYTHON_FALLTHROUGH [[fallthrough]]
    #elif __has_cpp_attribute(clang::fallthrough)
      #define CYTHON_FALLTHROUGH [[clang::fallthrough]]
    #elif __has_cpp_attribute(gnu::fallthrough)
      #define CYTHON_FALLTHROUGH [[gnu::fallthrough]]
    #endif
  #endif
  #ifndef CYTHON_FALLTHROUGH
    #if __has_attribute(fallthrough)
      #define CYTHON_FALLTHROUGH __attribute__((fallthrough))
    #else
      #define CYTHON_FALLTHROUGH
    #endif
  #endif
  #if defined(__clang__ ) && defined(__apple_build_version__)
    #if __apple_build_version__ < 7000000
      #undef  CYTHON_FALLTHROUGH
      #define CYTHON_FALLTHROUGH
    #endif
  #endif
#endif

#ifndef CYTHON_INLINE
  #if defined(__clang__)
    #define CYTHON_INLINE __inline__ __attribute__ ((__unused__))
  #elif defined(__GNUC__)
    #define CYTHON_INLINE __inline__
  #elif defined(_MSC_VER)
    #define CYTHON_INLINE __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_INLINE inline
  #else
    #define CYTHON_INLINE
  #endif
#endif

#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX < 0x02070600 && !defined(Py_OptimizeFlag)
  #define Py_OptimizeFlag 0
#endif
#define __PYX_BUILD_PY_SSIZE_T "n"
#define CYTHON_FORMAT_SSIZE_T "z"
#if PY_MAJOR_VERSION < 3
  #define __Pyx_BUILTIN_MODULE_NAME "__builtin__"
  #define __Pyx_PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)\
          PyCode_New(a+k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)
  #define __Pyx_DefaultClassType PyClass_Type
#else
  #define __Pyx_BUILTIN_MODULE_NAME "builtins"
  #define __Pyx_PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)\
          PyCode_New(a, k, l, s, f, code, c, n, v, fv, cell, fn, name, fline, lnos)
  #define __Pyx_DefaultClassType PyType_Type
#endif
#ifndef Py_TPFLAGS_CHECKTYPES
  #define Py_TPFLAGS_CHECKTYPES 0
#endif
#ifndef Py_TPFLAGS_HAVE_INDEX
  #define Py_TPFLAGS_HAVE_INDEX 0
#endif
#ifndef Py_TPFLAGS_HAVE_NEWBUFFER
  #define Py_TPFLAGS_HAVE_NEWBUFFER 0
#endif
#ifndef Py_TPFLAGS_HAVE_FINALIZE
  #define Py_TPFLAGS_HAVE_FINALIZE 0
#endif
#if PY_VERSION_HEX <= 0x030700A3 || !defined(METH_FASTCALL)
  #ifndef METH_FASTCALL
     #define METH_FASTCALL 0x80
  #endif
  typedef PyObject *(*__Pyx_PyCFunctionFast) (PyObject *self, PyObject *const *args, Py_ssize_t nargs);
  typedef PyObject *(*__Pyx_PyCFunctionFastWithKeywords) (PyObject *self, PyObject *const *args,
                                                          Py_ssize_t nargs, PyObject *kwnames);
#else
  #define __Pyx_PyCFunctionFast _PyCFunctionFast
  #define __Pyx_PyCFunctionFastWithKeywords _PyCFunctionFastWithKeywords
#endif
#if CYTHON_FAST_PYCCALL
#define __Pyx_PyFastCFunction_Check(func)\
    ((PyCFunction_Check(func) && (METH_FASTCALL == (PyCFunction_GET_FLAGS(func) & ~(METH_CLASS | METH_STATIC | METH_COEXIST | METH_KEYWORDS)))))
#else
#define __Pyx_PyFastCFunction_Check(func) 0
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyObject_Malloc)
  #define PyObject_Malloc(s)   PyMem_Malloc(s)
  #define PyObject_Free(p)     PyMem_Free(p)
  #define PyObject_Realloc(p)  PyMem_Realloc(p)
#endif
#if CYTHON_COMPILING_IN_PYSTON
  #define __Pyx_PyCode_HasFreeVars(co)  PyCode_HasFreeVars(co)
  #define __Pyx_PyFrame_SetLineNumber(frame, lineno) PyFrame_SetLineNumber(frame, lineno)
#else
  #define __Pyx_PyCode_HasFreeVars(co)  (PyCode_GetNumFree(co) > 0)
  #define __Pyx_PyFrame_SetLineNumber(frame, lineno)  (frame)->f_lineno = (lineno)
#endif
#if !CYTHON_FAST_THREAD_STATE || PY_VERSION_HEX < 0x02070000
  #define __Pyx_PyThreadState_Current PyThreadState_GET()
#elif PY_VERSION_HEX >= 0x03060000
  #define __Pyx_PyThreadState_Current _PyThreadState_UncheckedGet()
#elif PY_VERSION_HEX >= 0x03000000
  #define __Pyx_PyThreadState_Current PyThreadState_GET()
#else
  #define __Pyx_PyThreadState_Current _PyThreadState_Current
#endif
#if PY_VERSION_HEX < 0x030700A2 && !defined(PyThread_tss_create) && !defined(Py_tss_NEEDS_INIT)
#include "pythread.h"
#define Py_tss_NEEDS_INIT 0
typedef int Py_tss_t;
static CYTHON_INLINE int PyThread_tss_create(Py_tss_t *key) {
  *key = PyThread_create_key();
  return 0; // PyThread_create_key reports success always
}
static CYTHON_INLINE Py_tss_t * PyThread_tss_alloc(void) {
  Py_tss_t *key = (Py_tss_t *)PyObject_Malloc(sizeof(Py_tss_t));
  *key = Py_tss_NEEDS_INIT;
  return key;
}
static CYTHON_INLINE void PyThread_tss_free(Py_tss_t *key) {
  PyObject_Free(key);
}
static CYTHON_INLINE int PyThread_tss_is_created(Py_tss_t *key) {
  return *key != Py_tss_NEEDS_INIT;
}
static CYTHON_INLINE void PyThread_tss_delete(Py_tss_t *key) {
  PyThread_delete_key(*key);
  *key = Py_tss_NEEDS_INIT;
}
static CYTHON_INLINE int PyThread_tss_set(Py_tss_t *key, void *value) {
  return PyThread_set_key_value(*key, value);
}
static CYTHON_INLINE void * PyThread_tss_get(Py_tss_t *key) {
  return PyThread_get_key_value(*key);
}
#endif // TSS (Thread Specific Storage) API
#if CYTHON_COMPILING_IN_CPYTHON || defined(_PyDict_NewPresized)
#define __Pyx_PyDict_NewPresized(n)  ((n <= 8) ? PyDict_New() : _PyDict_NewPresized(n))
#else
#define __Pyx_PyDict_NewPresized(n)  PyDict_New()
#endif
#if PY_MAJOR_VERSION >= 3 || CYTHON_FUTURE_DIVISION
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_TrueDivide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceTrueDivide(x,y)
#else
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_Divide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceDivide(x,y)
#endif
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030500A1 && CYTHON_USE_UNICODE_INTERNALS
#define __Pyx_PyDict_GetItemStr(dict, name)  _PyDict_GetItem_KnownHash(dict, name, ((PyASCIIObject *) name)->hash)
#else
#define __Pyx_PyDict_GetItemStr(dict, name)  PyDict_GetItem(dict, name)
#endif
#if PY_VERSION_HEX > 0x03030000 && defined(PyUnicode_KIND)
  #define CYTHON_PEP393_ENABLED 1
  #define __Pyx_PyUnicode_READY(op)       (likely(PyUnicode_IS_READY(op)) ?\
                                              0 : _PyUnicode_Ready((PyObject *)(op)))
  #define __Pyx_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_LENGTH(u)
  #define __Pyx_PyUnicode_READ_CHAR(u, i) PyUnicode_READ_CHAR(u, i)
  #define __Pyx_PyUnicode_MAX_CHAR_VALUE(u)   PyUnicode_MAX_CHAR_VALUE(u)
  #define __Pyx_PyUnicode_KIND(u)         PyUnicode_KIND(u)
  #define __Pyx_PyUnicode_DATA(u)         PyUnicode_DATA(u)
  #define __Pyx_PyUnicode_READ(k, d, i)   PyUnicode_READ(k, d, i)
  #define __Pyx_PyUnicode_WRITE(k, d, i, ch)  PyUnicode_WRITE(k, d, i, ch)
  #define __Pyx_PyUnicode_IS_TRUE(u)      (0 != (likely(PyUnicode_IS_READY(u)) ? PyUnicode_GET_LENGTH(u) : PyUnicode_GET_SIZE(u)))
#else
  #define CYTHON_PEP393_ENABLED 0
  #define PyUnicode_1BYTE_KIND  1
  #define PyUnicode_2BYTE_KIND  2
  #define PyUnicode_4BYTE_KIND  4
  #define __Pyx_PyUnicode_READY(op)       (0)
  #define __Pyx_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_SIZE(u)
  #define __Pyx_PyUnicode_READ_CHAR(u, i) ((Py_UCS4)(PyUnicode_AS_UNICODE(u)[i]))
  #define __Pyx_PyUnicode_MAX_CHAR_VALUE(u)   ((sizeof(Py_UNICODE) == 2) ? 65535 : 1114111)
  #define __Pyx_PyUnicode_KIND(u)         (sizeof(Py_UNICODE))
  #define __Pyx_PyUnicode_DATA(u)         ((void*)PyUnicode_AS_UNICODE(u))
  #define __Pyx_PyUnicode_READ(k, d, i)   ((void)(k), (Py_UCS4)(((Py_UNICODE*)d)[i]))
  #define __Pyx_PyUnicode_WRITE(k, d, i, ch)  (((void)(k)), ((Py_UNICODE*)d)[i] = ch)
  #define __Pyx_PyUnicode_IS_TRUE(u)      (0 != PyUnicode_GET_SIZE(u))
#endif
#if CYTHON_COMPILING_IN_PYPY
  #define __Pyx_PyUnicode_Concat(a, b)      PyNumber_Add(a, b)
  #define __Pyx_PyUnicode_ConcatSafe(a, b)  PyNumber_Add(a, b)
#else
  #define __Pyx_PyUnicode_Concat(a, b)      PyUnicode_Concat(a, b)
  #define __Pyx_PyUnicode_ConcatSafe(a, b)  ((unlikely((a) == Py_None) || unlikely((b) == Py_None)) ?\
      PyNumber_Add(a, b) : __Pyx_PyUnicode_Concat(a, b))
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyUnicode_Contains)
  #define PyUnicode_Contains(u, s)  PySequence_Contains(u, s)
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyByteArray_Check)
  #define PyByteArray_Check(obj)  PyObject_TypeCheck(obj, &PyByteArray_Type)
#endif
#if CYTHON_COMPILING_IN_PYPY && !defined(PyObject_Format)
  #define PyObject_Format(obj, fmt)  PyObject_CallMethod(obj, "__format__", "O", fmt)
#endif
#define __Pyx_PyString_FormatSafe(a, b)   ((unlikely((a) == Py_None)) ? PyNumber_Remainder(a, b) : __Pyx_PyString_Format(a, b))
#define __Pyx_PyUnicode_FormatSafe(a, b)  ((unlikely((a) == Py_None)) ? PyNumber_Remainder(a, b) : PyUnicode_Format(a, b))
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyString_Format(a, b)  PyUnicode_Format(a, b)
#else
  #define __Pyx_PyString_Format(a, b)  PyString_Format(a, b)
#endif
#if PY_MAJOR_VERSION < 3 && !defined(PyObject_ASCII)
  #define PyObject_ASCII(o)            PyObject_Repr(o)
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyBaseString_Type            PyUnicode_Type
  #define PyStringObject               PyUnicodeObject
  #define PyString_Type                PyUnicode_Type
  #define PyString_Check               PyUnicode_Check
  #define PyString_CheckExact          PyUnicode_CheckExact
  #define PyObject_Unicode             PyObject_Str
#endif
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyBaseString_Check(obj) PyUnicode_Check(obj)
  #define __Pyx_PyBaseString_CheckExact(obj) PyUnicode_CheckExact(obj)
#else
  #define __Pyx_PyBaseString_Check(obj) (PyString_Check(obj) || PyUnicode_Check(obj))
  #define __Pyx_PyBaseString_CheckExact(obj) (PyString_CheckExact(obj) || PyUnicode_CheckExact(obj))
#endif
#ifndef PySet_CheckExact
  #define PySet_CheckExact(obj)        (Py_TYPE(obj) == &PySet_Type)
#endif
#if CYTHON_ASSUME_SAFE_MACROS
  #define __Pyx_PySequence_SIZE(seq)  Py_SIZE(seq)
#else
  #define __Pyx_PySequence_SIZE(seq)  PySequence_Size(seq)
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyIntObject                  PyLongObject
  #define PyInt_Type                   PyLong_Type
  #define PyInt_Check(op)              PyLong_Check(op)
  #define PyInt_CheckExact(op)         PyLong_CheckExact(op)
  #define PyInt_FromString             PyLong_FromString
  #define PyInt_FromUnicode            PyLong_FromUnicode
  #define PyInt_FromLong               PyLong_FromLong
  #define PyInt_FromSize_t             PyLong_FromSize_t
  #define PyInt_FromSsize_t            PyLong_FromSsize_t
  #define PyInt_AsLong                 PyLong_AsLong
  #define PyInt_AS_LONG                PyLong_AS_LONG
  #define PyInt_AsSsize_t              PyLong_AsSsize_t
  #define PyInt_AsUnsignedLongMask     PyLong_AsUnsignedLongMask
  #define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
  #define PyNumber_Int                 PyNumber_Long
#endif
#if PY_MAJOR_VERSION >= 3
  #define PyBoolObject                 PyLongObject
#endif
#if PY_MAJOR_VERSION >= 3 && CYTHON_COMPILING_IN_PYPY
  #ifndef PyUnicode_InternFromString
    #define PyUnicode_InternFromString(s) PyUnicode_FromString(s)
  #endif
#endif
#if PY_VERSION_HEX < 0x030200A4
  typedef long Py_hash_t;
  #define __Pyx_PyInt_FromHash_t PyInt_FromLong
  #define __Pyx_PyInt_AsHash_t   PyInt_AsLong
#else
  #define __Pyx_PyInt_FromHash_t PyInt_FromSsize_t
  #define __Pyx_PyInt_AsHash_t   PyInt_AsSsize_t
#endif
#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyMethod_New(func, self, klass) ((self) ? PyMethod_New(func, self) : (Py_INCREF(func), func))
#else
  #define __Pyx_PyMethod_New(func, self, klass) PyMethod_New(func, self, klass)
#endif
#if CYTHON_USE_ASYNC_SLOTS
  #if PY_VERSION_HEX >= 0x030500B1
    #define __Pyx_PyAsyncMethodsStruct PyAsyncMethods
    #define __Pyx_PyType_AsAsync(obj) (Py_TYPE(obj)->tp_as_async)
  #else
    #define __Pyx_PyType_AsAsync(obj) ((__Pyx_PyAsyncMethodsStruct*) (Py_TYPE(obj)->tp_reserved))
  #endif
#else
  #define __Pyx_PyType_AsAsync(obj) NULL
#endif
#ifndef __Pyx_PyAsyncMethodsStruct
    typedef struct {
        unaryfunc am_await;
        unaryfunc am_aiter;
        unaryfunc am_anext;
    } __Pyx_PyAsyncMethodsStruct;
#endif

#if defined(WIN32) || defined(MS_WINDOWS)
  #define _USE_MATH_DEFINES
#endif
#include <math.h>
#ifdef NAN
#define __PYX_NAN() ((float) NAN)
#else
static CYTHON_INLINE float __PYX_NAN() {
  float value;
  memset(&value, 0xFF, sizeof(value));
  return value;
}
#endif
#if defined(__CYGWIN__) && defined(_LDBL_EQ_DBL)
#define __Pyx_truncl trunc
#else
#define __Pyx_truncl truncl
#endif


#define __PYX_ERR(f_index, lineno, Ln_error) \
{ \
  __pyx_filename = __pyx_f[f_index]; __pyx_lineno = lineno; __pyx_clineno = __LINE__; goto Ln_error; \
}

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#define __PYX_HAVE__crowdposetools___mask
#define __PYX_HAVE_API__crowdposetools___mask
/* Early includes */
#include <string.h>
#include <stdio.h>
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include <stdlib.h>
#include "maskApi.h"
#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#if defined(PYREX_WITHOUT_ASSERTIONS) && !defined(CYTHON_WITHOUT_ASSERTIONS)
#define CYTHON_WITHOUT_ASSERTIONS
#endif

typedef struct {PyObject **p; const char *s; const Py_ssize_t n; const char* encoding;
                const char is_unicode; const char is_str; const char intern; } __Pyx_StringTabEntry;

#define __PYX_DEFAULT_STRING_ENCODING_IS_ASCII 0
#define __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT 0
#define __PYX_DEFAULT_STRING_ENCODING ""
#define __Pyx_PyObject_FromString __Pyx_PyBytes_FromString
#define __Pyx_PyObject_FromStringAndSize __Pyx_PyBytes_FromStringAndSize
#define __Pyx_uchar_cast(c) ((unsigned char)c)
#define __Pyx_long_cast(x) ((long)x)
#define __Pyx_fits_Py_ssize_t(v, type, is_signed)  (\
    (sizeof(type) < sizeof(Py_ssize_t))  ||\
    (sizeof(type) > sizeof(Py_ssize_t) &&\
          likely(v < (type)PY_SSIZE_T_MAX ||\
                 v == (type)PY_SSIZE_T_MAX)  &&\
          (!is_signed || likely(v > (type)PY_SSIZE_T_MIN ||\
                                v == (type)PY_SSIZE_T_MIN)))  ||\
    (sizeof(type) == sizeof(Py_ssize_t) &&\
          (is_signed || likely(v < (type)PY_SSIZE_T_MAX ||\
                               v == (type)PY_SSIZE_T_MAX)))  )
#if defined (__cplusplus) && __cplusplus >= 201103L
    #include <cstdlib>
    #define __Pyx_sst_abs(value) std::abs(value)
#elif SIZEOF_INT >= SIZEOF_SIZE_T
    #define __Pyx_sst_abs(value) abs(value)
#elif SIZEOF_LONG >= SIZEOF_SIZE_T
    #define __Pyx_sst_abs(value) labs(value)
#elif defined (_MSC_VER)
    #define __Pyx_sst_abs(value) ((Py_ssize_t)_abs64(value))
#elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define __Pyx_sst_abs(value) llabs(value)
#elif defined (__GNUC__)
    #define __Pyx_sst_abs(value) __builtin_llabs(value)
#else
    #define __Pyx_sst_abs(value) ((value<0) ? -value : value)
#endif
static CYTHON_INLINE const char* __Pyx_PyObject_AsString(PyObject*);
static CYTHON_INLINE const char* __Pyx_PyObject_AsStringAndSize(PyObject*, Py_ssize_t* length);
#define __Pyx_PyByteArray_FromString(s) PyByteArray_FromStringAndSize((const char*)s, strlen((const char*)s))
#define __Pyx_PyByteArray_FromStringAndSize(s, l) PyByteArray_FromStringAndSize((const char*)s, l)
#define __Pyx_PyBytes_FromString        PyBytes_FromString
#define __Pyx_PyBytes_FromStringAndSize PyBytes_FromStringAndSize
static CYTHON_INLINE PyObject* __Pyx_PyUnicode_FromString(const char*);
#if PY_MAJOR_VERSION < 3
    #define __Pyx_PyStr_FromString        __Pyx_PyBytes_FromString
    #define __Pyx_PyStr_FromStringAndSize __Pyx_PyBytes_FromStringAndSize
#else
    #define __Pyx_PyStr_FromString        __Pyx_PyUnicode_FromString
    #define __Pyx_PyStr_FromStringAndSize __Pyx_PyUnicode_FromStringAndSize
#endif
#define __Pyx_PyBytes_AsWritableString(s)     ((char*) PyBytes_AS_STRING(s))
#define __Pyx_PyBytes_AsWritableSString(s)    ((signed char*) PyBytes_AS_STRING(s))
#define __Pyx_PyBytes_AsWritableUString(s)    ((unsigned char*) PyBytes_AS_STRING(s))
#define __Pyx_PyBytes_AsString(s)     ((const char*) PyBytes_AS_STRING(s))
#define __Pyx_PyBytes_AsSString(s)    ((const signed char*) PyBytes_AS_STRING(s))
#define __Pyx_PyBytes_AsUString(s)    ((const unsigned char*) PyBytes_AS_STRING(s))
#define __Pyx_PyObject_AsWritableString(s)    ((char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_AsWritableSString(s)    ((signed char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_AsWritableUString(s)    ((unsigned char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_AsSString(s)    ((const signed char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_AsUString(s)    ((const unsigned char*) __Pyx_PyObject_AsString(s))
#define __Pyx_PyObject_FromCString(s)  __Pyx_PyObject_FromString((const char*)s)
#define __Pyx_PyBytes_FromCString(s)   __Pyx_PyBytes_FromString((const char*)s)
#define __Pyx_PyByteArray_FromCString(s)   __Pyx_PyByteArray_FromString((const char*)s)
#define __Pyx_PyStr_FromCString(s)     __Pyx_PyStr_FromString((const char*)s)
#define __Pyx_PyUnicode_FromCString(s) __Pyx_PyUnicode_FromString((const char*)s)
static CYTHON_INLINE size_t __Pyx_Py_UNICODE_strlen(const Py_UNICODE *u) {
    const Py_UNICODE *u_end = u;
    while (*u_end++) ;
    return (size_t)(u_end - u - 1);
}
#define __Pyx_PyUnicode_FromUnicode(u)       PyUnicode_FromUnicode(u, __Pyx_Py_UNICODE_strlen(u))
#define __Pyx_PyUnicode_FromUnicodeAndLength PyUnicode_FromUnicode
#define __Pyx_PyUnicode_AsUnicode            PyUnicode_AsUnicode
#define __Pyx_NewRef(obj) (Py_INCREF(obj), obj)
#define __Pyx_Owned_Py_None(b) __Pyx_NewRef(Py_None)
static CYTHON_INLINE PyObject * __Pyx_PyBool_FromLong(long b);
static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject*);
static CYTHON_INLINE PyObject* __Pyx_PyNumber_IntOrLong(PyObject* x);
#define __Pyx_PySequence_Tuple(obj)\
    (likely(PyTuple_CheckExact(obj)) ? __Pyx_NewRef(obj) : PySequence_Tuple(obj))
static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject*);
static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t);
#if CYTHON_ASSUME_SAFE_MACROS
#define __pyx_PyFloat_AsDouble(x) (PyFloat_CheckExact(x) ? PyFloat_AS_DOUBLE(x) : PyFloat_AsDouble(x))
#else
#define __pyx_PyFloat_AsDouble(x) PyFloat_AsDouble(x)
#endif
#define __pyx_PyFloat_AsFloat(x) ((float) __pyx_PyFloat_AsDouble(x))
#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyNumber_Int(x) (PyLong_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Long(x))
#else
#define __Pyx_PyNumber_Int(x) (PyInt_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Int(x))
#endif
#define __Pyx_PyNumber_Float(x) (PyFloat_CheckExact(x) ? __Pyx_NewRef(x) : PyNumber_Float(x))
#if PY_MAJOR_VERSION < 3 && __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
static int __Pyx_sys_getdefaultencoding_not_ascii;
static int __Pyx_init_sys_getdefaultencoding_params(void) {
    PyObject* sys;
    PyObject* default_encoding = NULL;
    PyObject* ascii_chars_u = NULL;
    PyObject* ascii_chars_b = NULL;
    const char* default_encoding_c;
    sys = PyImport_ImportModule("sys");
    if (!sys) goto bad;
    default_encoding = PyObject_CallMethod(sys, (char*) "getdefaultencoding", NULL);
    Py_DECREF(sys);
    if (!default_encoding) goto bad;
    default_encoding_c = PyBytes_AsString(default_encoding);
    if (!default_encoding_c) goto bad;
    if (strcmp(default_encoding_c, "ascii") == 0) {
        __Pyx_sys_getdefaultencoding_not_ascii = 0;
    } else {
        char ascii_chars[128];
        int c;
        for (c = 0; c < 128; c++) {
            ascii_chars[c] = c;
        }
        __Pyx_sys_getdefaultencoding_not_ascii = 1;
        ascii_chars_u = PyUnicode_DecodeASCII(ascii_chars, 128, NULL);
        if (!ascii_chars_u) goto bad;
        ascii_chars_b = PyUnicode_AsEncodedString(ascii_chars_u, default_encoding_c, NULL);
        if (!ascii_chars_b || !PyBytes_Check(ascii_chars_b) || memcmp(ascii_chars, PyBytes_AS_STRING(ascii_chars_b), 128) != 0) {
            PyErr_Format(
                PyExc_ValueError,
                "This module compiled with c_string_encoding=ascii, but default encoding '%.200s' is not a superset of ascii.",
                default_encoding_c);
            goto bad;
        }
        Py_DECREF(ascii_chars_u);
        Py_DECREF(ascii_chars_b);
    }
    Py_DECREF(default_encoding);
    return 0;
bad:
    Py_XDECREF(default_encoding);
    Py_XDECREF(ascii_chars_u);
    Py_XDECREF(ascii_chars_b);
    return -1;
}
#endif
#if __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT && PY_MAJOR_VERSION >= 3
#define __Pyx_PyUnicode_FromStringAndSize(c_str, size) PyUnicode_DecodeUTF8(c_str, size, NULL)
#else
#define __Pyx_PyUnicode_FromStringAndSize(c_str, size) PyUnicode_Decode(c_str, size, __PYX_DEFAULT_STRING_ENCODING, NULL)
#if __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT
static char* __PYX_DEFAULT_STRING_ENCODING;
static int __Pyx_init_sys_getdefaultencoding_params(void) {
    PyObject* sys;
    PyObject* default_encoding = NULL;
    char* default_encoding_c;
    sys = PyImport_ImportModule("sys");
    if (!sys) goto bad;
    default_encoding = PyObject_CallMethod(sys, (char*) (const char*) "getdefaultencoding", NULL);
    Py_DECREF(sys);
    if (!default_encoding) goto bad;
    default_encoding_c = PyBytes_AsString(default_encoding);
    if (!default_encoding_c) goto bad;
    __PYX_DEFAULT_STRING_ENCODING = (char*) malloc(strlen(default_encoding_c));
    if (!__PYX_DEFAULT_STRING_ENCODING) goto bad;
    strcpy(__PYX_DEFAULT_STRING_ENCODING, default_encoding_c);
    Py_DECREF(default_encoding);
    return 0;
bad:
    Py_XDECREF(default_encoding);
    return -1;
}
#endif
#endif


/* Test for GCC > 2.95 */
#if defined(__GNUC__)     && (__GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95)))
  #define likely(x)   __builtin_expect(!!(x), 1)
  #define unlikely(x) __builtin_expect(!!(x), 0)
#else /* !__GNUC__ or GCC < 2.95 */
  #define likely(x)   (x)
  #define unlikely(x) (x)
#endif /* __GNUC__ */
static CYTHON_INLINE void __Pyx_pretend_to_initialize(void* ptr) { (void)ptr; }

static PyObject *__pyx_m = NULL;
static PyObject *__pyx_d;
static PyObject *__pyx_b;
static PyObject *__pyx_cython_runtime = NULL;
static PyObject *__pyx_empty_tuple;
static PyObject *__pyx_empty_bytes;
static PyObject *__pyx_empty_unicode;
static int __pyx_lineno;
static int __pyx_clineno = 0;
static const char * __pyx_cfilenm= __FILE__;
static const char *__pyx_filename;

/* Header.proto */
#if !defined(CYTHON_CCOMPLEX)
  #if defined(__cplusplus)
    #define CYTHON_CCOMPLEX 1
  #elif defined(_Complex_I)
    #define CYTHON_CCOMPLEX 1
  #else
    #define CYTHON_CCOMPLEX 0
  #endif
#endif
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #include <complex>
  #else
    #include <complex.h>
  #endif
#endif
#if CYTHON_CCOMPLEX && !defined(__cplusplus) && defined(__sun__) && defined(__GNUC__)
  #undef _Complex_I
  #define _Complex_I 1.0fj
#endif


static const char *__pyx_f[] = {
  "crowdposetools/_mask.pyx",
  "stringsource",
  "__init__.pxd",
  "type.pxd",
};
/* BufferFormatStructs.proto */
#define IS_UNSIGNED(type) (((type) -1) > 0)
struct __Pyx_StructField_;
#define __PYX_BUF_FLAGS_PACKED_STRUCT (1 << 0)
typedef struct {
  const char* name;
  struct __Pyx_StructField_* fields;
  size_t size;
  size_t arraysize[8];
  int ndim;
  char typegroup;
  char is_unsigned;
  int flags;
} __Pyx_TypeInfo;
typedef struct __Pyx_StructField_ {
  __Pyx_TypeInfo* type;
  const char* name;
  size_t offset;
} __Pyx_StructField;
typedef struct {
  __Pyx_StructField* field;
  size_t parent_offset;
} __Pyx_BufFmt_StackElem;
typedef struct {
  __Pyx_StructField root;
  __Pyx_BufFmt_StackElem* head;
  size_t fmt_offset;
  size_t new_count, enc_count;
  size_t struct_alignment;
  int is_complex;
  char enc_type;
  char new_packmode;
  char enc_packmode;
  char is_valid_array;
} __Pyx_BufFmt_Context;


/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":730
 * # in Cython to enable them only on the right systems.
 * 
 * ctypedef npy_int8       int8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 */
typedef npy_int8 __pyx_t_5numpy_int8_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":731
 * 
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t
 */
typedef npy_int16 __pyx_t_5numpy_int16_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":732
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int64      int64_t
 * #ctypedef npy_int96      int96_t
 */
typedef npy_int32 __pyx_t_5numpy_int32_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":733
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_int96      int96_t
 * #ctypedef npy_int128     int128_t
 */
typedef npy_int64 __pyx_t_5numpy_int64_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":737
 * #ctypedef npy_int128     int128_t
 * 
 * ctypedef npy_uint8      uint8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 */
typedef npy_uint8 __pyx_t_5numpy_uint8_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":738
 * 
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t
 */
typedef npy_uint16 __pyx_t_5numpy_uint16_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":739
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint64     uint64_t
 * #ctypedef npy_uint96     uint96_t
 */
typedef npy_uint32 __pyx_t_5numpy_uint32_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":740
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_uint96     uint96_t
 * #ctypedef npy_uint128    uint128_t
 */
typedef npy_uint64 __pyx_t_5numpy_uint64_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":744
 * #ctypedef npy_uint128    uint128_t
 * 
 * ctypedef npy_float32    float32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_float64    float64_t
 * #ctypedef npy_float80    float80_t
 */
typedef npy_float32 __pyx_t_5numpy_float32_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":745
 * 
 * ctypedef npy_float32    float32_t
 * ctypedef npy_float64    float64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_float80    float80_t
 * #ctypedef npy_float128   float128_t
 */
typedef npy_float64 __pyx_t_5numpy_float64_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":754
 * # The int types are mapped a bit surprising --
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t
 */
typedef npy_long __pyx_t_5numpy_int_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":755
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   longlong_t
 * 
 */
typedef npy_longlong __pyx_t_5numpy_long_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":756
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_ulong      uint_t
 */
typedef npy_longlong __pyx_t_5numpy_longlong_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":758
 * ctypedef npy_longlong   longlong_t
 * 
 * ctypedef npy_ulong      uint_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t
 */
typedef npy_ulong __pyx_t_5numpy_uint_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":759
 * 
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 */
typedef npy_ulonglong __pyx_t_5numpy_ulong_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":760
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_intp       intp_t
 */
typedef npy_ulonglong __pyx_t_5numpy_ulonglong_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":762
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 * ctypedef npy_intp       intp_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uintp      uintp_t
 * 
 */
typedef npy_intp __pyx_t_5numpy_intp_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":763
 * 
 * ctypedef npy_intp       intp_t
 * ctypedef npy_uintp      uintp_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_double     float_t
 */
typedef npy_uintp __pyx_t_5numpy_uintp_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":765
 * ctypedef npy_uintp      uintp_t
 * 
 * ctypedef npy_double     float_t             # <<<<<<<<<<<<<<
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t
 */
typedef npy_double __pyx_t_5numpy_float_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":766
 * 
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longdouble longdouble_t
 * 
 */
typedef npy_double __pyx_t_5numpy_double_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":767
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cfloat      cfloat_t
 */
typedef npy_longdouble __pyx_t_5numpy_longdouble_t;
/* Declarations.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< float > __pyx_t_float_complex;
  #else
    typedef float _Complex __pyx_t_float_complex;
  #endif
#else
    typedef struct { float real, imag; } __pyx_t_float_complex;
#endif
static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float, float);

/* Declarations.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< double > __pyx_t_double_complex;
  #else
    typedef double _Complex __pyx_t_double_complex;
  #endif
#else
    typedef struct { double real, imag; } __pyx_t_double_complex;
#endif
static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double, double);


/*--- Type declarations ---*/
struct __pyx_obj_14crowdposetools_5_mask_RLEs;
struct __pyx_obj_14crowdposetools_5_mask_Masks;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":769
 * ctypedef npy_longdouble longdouble_t
 * 
 * ctypedef npy_cfloat      cfloat_t             # <<<<<<<<<<<<<<
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t
 */
typedef npy_cfloat __pyx_t_5numpy_cfloat_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":770
 * 
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t             # <<<<<<<<<<<<<<
 * ctypedef npy_clongdouble clongdouble_t
 * 
 */
typedef npy_cdouble __pyx_t_5numpy_cdouble_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":771
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cdouble     complex_t
 */
typedef npy_clongdouble __pyx_t_5numpy_clongdouble_t;

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":773
 * ctypedef npy_clongdouble clongdouble_t
 * 
 * ctypedef npy_cdouble     complex_t             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 */
typedef npy_cdouble __pyx_t_5numpy_complex_t;

/* "crowdposetools/_mask.pyx":56
 * # python class to wrap RLE array in C
 * # the class handles the memory allocation and deallocation
 * cdef class RLEs:             # <<<<<<<<<<<<<<
 *     cdef RLE *_R
 *     cdef siz _n
 */
struct __pyx_obj_14crowdposetools_5_mask_RLEs {
  PyObject_HEAD
  RLE *_R;
  siz _n;
};


/* "crowdposetools/_mask.pyx":77
 * # python class to wrap Mask array in C
 * # the class handles the memory allocation and deallocation
 * cdef class Masks:             # <<<<<<<<<<<<<<
 *     cdef byte *_mask
 *     cdef siz _h
 */
struct __pyx_obj_14crowdposetools_5_mask_Masks {
  PyObject_HEAD
  byte *_mask;
  siz _h;
  siz _w;
  siz _n;
};


/* --- Runtime support code (head) --- */
/* Refnanny.proto */
#ifndef CYTHON_REFNANNY
  #define CYTHON_REFNANNY 0
#endif
#if CYTHON_REFNANNY
  typedef struct {
    void (*INCREF)(void*, PyObject*, int);
    void (*DECREF)(void*, PyObject*, int);
    void (*GOTREF)(void*, PyObject*, int);
    void (*GIVEREF)(void*, PyObject*, int);
    void* (*SetupContext)(const char*, int, const char*);
    void (*FinishContext)(void**);
  } __Pyx_RefNannyAPIStruct;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNanny = NULL;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname);
  #define __Pyx_RefNannyDeclarations void *__pyx_refnanny = NULL;
#ifdef WITH_THREAD
  #define __Pyx_RefNannySetupContext(name, acquire_gil)\
          if (acquire_gil) {\
              PyGILState_STATE __pyx_gilstate_save = PyGILState_Ensure();\
              __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__);\
              PyGILState_Release(__pyx_gilstate_save);\
          } else {\
              __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__);\
          }
#else
  #define __Pyx_RefNannySetupContext(name, acquire_gil)\
          __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__)
#endif
  #define __Pyx_RefNannyFinishContext()\
          __Pyx_RefNanny->FinishContext(&__pyx_refnanny)
  #define __Pyx_INCREF(r)  __Pyx_RefNanny->INCREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_DECREF(r)  __Pyx_RefNanny->DECREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GOTREF(r)  __Pyx_RefNanny->GOTREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GIVEREF(r) __Pyx_RefNanny->GIVEREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_XINCREF(r)  do { if((r) != NULL) {__Pyx_INCREF(r); }} while(0)
  #define __Pyx_XDECREF(r)  do { if((r) != NULL) {__Pyx_DECREF(r); }} while(0)
  #define __Pyx_XGOTREF(r)  do { if((r) != NULL) {__Pyx_GOTREF(r); }} while(0)
  #define __Pyx_XGIVEREF(r) do { if((r) != NULL) {__Pyx_GIVEREF(r);}} while(0)
#else
  #define __Pyx_RefNannyDeclarations
  #define __Pyx_RefNannySetupContext(name, acquire_gil)
  #define __Pyx_RefNannyFinishContext()
  #define __Pyx_INCREF(r) Py_INCREF(r)
  #define __Pyx_DECREF(r) Py_DECREF(r)
  #define __Pyx_GOTREF(r)
  #define __Pyx_GIVEREF(r)
  #define __Pyx_XINCREF(r) Py_XINCREF(r)
  #define __Pyx_XDECREF(r) Py_XDECREF(r)
  #define __Pyx_XGOTREF(r)
  #define __Pyx_XGIVEREF(r)
#endif
#define __Pyx_XDECREF_SET(r, v) do {\
        PyObject *tmp = (PyObject *) r;\
        r = v; __Pyx_XDECREF(tmp);\
    } while (0)
#define __Pyx_DECREF_SET(r, v) do {\
        PyObject *tmp = (PyObject *) r;\
        r = v; __Pyx_DECREF(tmp);\
    } while (0)
#define __Pyx_CLEAR(r)    do { PyObject* tmp = ((PyObject*)(r)); r = NULL; __Pyx_DECREF(tmp);} while(0)
#define __Pyx_XCLEAR(r)   do { if((r) != NULL) {PyObject* tmp = ((PyObject*)(r)); r = NULL; __Pyx_DECREF(tmp);}} while(0)

/* PyObjectGetAttrStr.proto */
#if CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStr(PyObject* obj, PyObject* attr_name);
#else
#define __Pyx_PyObject_GetAttrStr(o,n) PyObject_GetAttr(o,n)
#endif

/* GetBuiltinName.proto */
static PyObject *__Pyx_GetBuiltinName(PyObject *name);

/* RaiseDoubleKeywords.proto */
static void __Pyx_RaiseDoubleKeywordsError(const char* func_name, PyObject* kw_name);

/* ParseKeywords.proto */
static int __Pyx_ParseOptionalKeywords(PyObject *kwds, PyObject **argnames[],\
    PyObject *kwds2, PyObject *values[], Py_ssize_t num_pos_args,\
    const char* function_name);

/* RaiseArgTupleInvalid.proto */
static void __Pyx_RaiseArgtupleInvalid(const char* func_name, int exact,
    Py_ssize_t num_min, Py_ssize_t num_max, Py_ssize_t num_found);

/* IncludeStringH.proto */
#include <string.h>

/* BytesEquals.proto */
static CYTHON_INLINE int __Pyx_PyBytes_Equals(PyObject* s1, PyObject* s2, int equals);

/* UnicodeEquals.proto */
static CYTHON_INLINE int __Pyx_PyUnicode_Equals(PyObject* s1, PyObject* s2, int equals);

/* StrEquals.proto */
#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyString_Equals __Pyx_PyUnicode_Equals
#else
#define __Pyx_PyString_Equals __Pyx_PyBytes_Equals
#endif

/* PyCFunctionFastCall.proto */
#if CYTHON_FAST_PYCCALL
static CYTHON_INLINE PyObject *__Pyx_PyCFunction_FastCall(PyObject *func, PyObject **args, Py_ssize_t nargs);
#else
#define __Pyx_PyCFunction_FastCall(func, args, nargs)  (assert(0), NULL)
#endif

/* PyFunctionFastCall.proto */
#if CYTHON_FAST_PYCALL
#define __Pyx_PyFunction_FastCall(func, args, nargs)\
    __Pyx_PyFunction_FastCallDict((func), (args), (nargs), NULL)
#if 1 || PY_VERSION_HEX < 0x030600B1
static PyObject *__Pyx_PyFunction_FastCallDict(PyObject *func, PyObject **args, int nargs, PyObject *kwargs);
#else
#define __Pyx_PyFunction_FastCallDict(func, args, nargs, kwargs) _PyFunction_FastCallDict(func, args, nargs, kwargs)
#endif
#endif

/* PyObjectCall.proto */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw);
#else
#define __Pyx_PyObject_Call(func, arg, kw) PyObject_Call(func, arg, kw)
#endif

/* PyObjectCallMethO.proto */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallMethO(PyObject *func, PyObject *arg);
#endif

/* PyObjectCallOneArg.proto */
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg);

/* PyThreadStateGet.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_PyThreadState_declare  PyThreadState *__pyx_tstate;
#define __Pyx_PyThreadState_assign  __pyx_tstate = __Pyx_PyThreadState_Current;
#define __Pyx_PyErr_Occurred()  __pyx_tstate->curexc_type
#else
#define __Pyx_PyThreadState_declare
#define __Pyx_PyThreadState_assign
#define __Pyx_PyErr_Occurred()  PyErr_Occurred()
#endif

/* PyErrFetchRestore.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_PyErr_Clear() __Pyx_ErrRestore(NULL, NULL, NULL)
#define __Pyx_ErrRestoreWithState(type, value, tb)  __Pyx_ErrRestoreInState(PyThreadState_GET(), type, value, tb)
#define __Pyx_ErrFetchWithState(type, value, tb)    __Pyx_ErrFetchInState(PyThreadState_GET(), type, value, tb)
#define __Pyx_ErrRestore(type, value, tb)  __Pyx_ErrRestoreInState(__pyx_tstate, type, value, tb)
#define __Pyx_ErrFetch(type, value, tb)    __Pyx_ErrFetchInState(__pyx_tstate, type, value, tb)
static CYTHON_INLINE void __Pyx_ErrRestoreInState(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb);
static CYTHON_INLINE void __Pyx_ErrFetchInState(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb);
#if CYTHON_COMPILING_IN_CPYTHON
#define __Pyx_PyErr_SetNone(exc) (Py_INCREF(exc), __Pyx_ErrRestore((exc), NULL, NULL))
#else
#define __Pyx_PyErr_SetNone(exc) PyErr_SetNone(exc)
#endif
#else
#define __Pyx_PyErr_Clear() PyErr_Clear()
#define __Pyx_PyErr_SetNone(exc) PyErr_SetNone(exc)
#define __Pyx_ErrRestoreWithState(type, value, tb)  PyErr_Restore(type, value, tb)
#define __Pyx_ErrFetchWithState(type, value, tb)  PyErr_Fetch(type, value, tb)
#define __Pyx_ErrRestoreInState(tstate, type, value, tb)  PyErr_Restore(type, value, tb)
#define __Pyx_ErrFetchInState(tstate, type, value, tb)  PyErr_Fetch(type, value, tb)
#define __Pyx_ErrRestore(type, value, tb)  PyErr_Restore(type, value, tb)
#define __Pyx_ErrFetch(type, value, tb)  PyErr_Fetch(type, value, tb)
#endif

/* RaiseException.proto */
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause);

/* ExtTypeTest.proto */
static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type);

/* ArgTypeTest.proto */
#define __Pyx_ArgTypeTest(obj, type, none_allowed, name, exact)\
    ((likely((Py_TYPE(obj) == type) | (none_allowed && (obj == Py_None)))) ? 1 :\
        __Pyx__ArgTypeTest(obj, type, name, exact))
static int __Pyx__ArgTypeTest(PyObject *obj, PyTypeObject *type, const char *name, int exact);

/* ListAppend.proto */
#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE int __Pyx_PyList_Append(PyObject* list, PyObject* x) {
    PyListObject* L = (PyListObject*) list;
    Py_ssize_t len = Py_SIZE(list);
    if (likely(L->allocated > len) & likely(len > (L->allocated >> 1))) {
        Py_INCREF(x);
        PyList_SET_ITEM(list, len, x);
        Py_SIZE(list) = len+1;
        return 0;
    }
    return PyList_Append(list, x);
}
#else
#define __Pyx_PyList_Append(L,x) PyList_Append(L,x)
#endif

/* PyIntBinop.proto */
#if !CYTHON_COMPILING_IN_PYPY
static PyObject* __Pyx_PyInt_AddObjC(PyObject *op1, PyObject *op2, long intval, int inplace);
#else
#define __Pyx_PyInt_AddObjC(op1, op2, intval, inplace)\
    (inplace ? PyNumber_InPlaceAdd(op1, op2) : PyNumber_Add(op1, op2))
#endif

/* PyIntBinop.proto */
#if !CYTHON_COMPILING_IN_PYPY
static PyObject* __Pyx_PyInt_EqObjC(PyObject *op1, PyObject *op2, long intval, int inplace);
#else
#define __Pyx_PyInt_EqObjC(op1, op2, intval, inplace)\
    PyObject_RichCompare(op1, op2, Py_EQ)
    #endif

/* GetModuleGlobalName.proto */
static CYTHON_INLINE PyObject *__Pyx_GetModuleGlobalName(PyObject *name);

/* DictGetItem.proto */
#if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
static PyObject *__Pyx_PyDict_GetItem(PyObject *d, PyObject* key);
#define __Pyx_PyObject_Dict_GetItem(obj, name)\
    (likely(PyDict_CheckExact(obj)) ?\
     __Pyx_PyDict_GetItem(obj, name) : PyObject_GetItem(obj, name))
#else
#define __Pyx_PyDict_GetItem(d, key) PyObject_GetItem(d, key)
#define __Pyx_PyObject_Dict_GetItem(obj, name)  PyObject_GetItem(obj, name)
#endif

/* GetItemInt.proto */
#define __Pyx_GetItemInt(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck)\
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ?\
    __Pyx_GetItemInt_Fast(o, (Py_ssize_t)i, is_list, wraparound, boundscheck) :\
    (is_list ? (PyErr_SetString(PyExc_IndexError, "list index out of range"), (PyObject*)NULL) :\
               __Pyx_GetItemInt_Generic(o, to_py_func(i))))
#define __Pyx_GetItemInt_List(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck)\
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ?\
    __Pyx_GetItemInt_List_Fast(o, (Py_ssize_t)i, wraparound, boundscheck) :\
    (PyErr_SetString(PyExc_IndexError, "list index out of range"), (PyObject*)NULL))
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_List_Fast(PyObject *o, Py_ssize_t i,
                                                              int wraparound, int boundscheck);
#define __Pyx_GetItemInt_Tuple(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck)\
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ?\
    __Pyx_GetItemInt_Tuple_Fast(o, (Py_ssize_t)i, wraparound, boundscheck) :\
    (PyErr_SetString(PyExc_IndexError, "tuple index out of range"), (PyObject*)NULL))
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_Tuple_Fast(PyObject *o, Py_ssize_t i,
                                                              int wraparound, int boundscheck);
static PyObject *__Pyx_GetItemInt_Generic(PyObject *o, PyObject* j);
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_Fast(PyObject *o, Py_ssize_t i,
                                                     int is_list, int wraparound, int boundscheck);

/* IsLittleEndian.proto */
static CYTHON_INLINE int __Pyx_Is_Little_Endian(void);

/* BufferFormatCheck.proto */
static const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts);
static void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type);

/* BufferGetAndValidate.proto */
#define __Pyx_GetBufferAndValidate(buf, obj, dtype, flags, nd, cast, stack)\
    ((obj == Py_None || obj == NULL) ?\
    (__Pyx_ZeroBuffer(buf), 0) :\
    __Pyx__GetBufferAndValidate(buf, obj, dtype, flags, nd, cast, stack))
static int  __Pyx__GetBufferAndValidate(Py_buffer* buf, PyObject* obj,
    __Pyx_TypeInfo* dtype, int flags, int nd, int cast, __Pyx_BufFmt_StackElem* stack);
static void __Pyx_ZeroBuffer(Py_buffer* buf);
static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info);
static Py_ssize_t __Pyx_minusones[] = { -1, -1, -1, -1, -1, -1, -1, -1 };
static Py_ssize_t __Pyx_zeros[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

/* ListCompAppend.proto */
#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE int __Pyx_ListComp_Append(PyObject* list, PyObject* x) {
    PyListObject* L = (PyListObject*) list;
    Py_ssize_t len = Py_SIZE(list);
    if (likely(L->allocated > len)) {
        Py_INCREF(x);
        PyList_SET_ITEM(list, len, x);
        Py_SIZE(list) = len+1;
        return 0;
    }
    return PyList_Append(list, x);
}
#else
#define __Pyx_ListComp_Append(L,x) PyList_Append(L,x)
#endif

/* FetchCommonType.proto */
static PyTypeObject* __Pyx_FetchCommonType(PyTypeObject* type);

/* CythonFunction.proto */
#define __Pyx_CyFunction_USED 1
#define __Pyx_CYFUNCTION_STATICMETHOD  0x01
#define __Pyx_CYFUNCTION_CLASSMETHOD   0x02
#define __Pyx_CYFUNCTION_CCLASS        0x04
#define __Pyx_CyFunction_GetClosure(f)\
    (((__pyx_CyFunctionObject *) (f))->func_closure)
#define __Pyx_CyFunction_GetClassObj(f)\
    (((__pyx_CyFunctionObject *) (f))->func_classobj)
#define __Pyx_CyFunction_Defaults(type, f)\
    ((type *)(((__pyx_CyFunctionObject *) (f))->defaults))
#define __Pyx_CyFunction_SetDefaultsGetter(f, g)\
    ((__pyx_CyFunctionObject *) (f))->defaults_getter = (g)
typedef struct {
    PyCFunctionObject func;
#if PY_VERSION_HEX < 0x030500A0
    PyObject *func_weakreflist;
#endif
    PyObject *func_dict;
    PyObject *func_name;
    PyObject *func_qualname;
    PyObject *func_doc;
    PyObject *func_globals;
    PyObject *func_code;
    PyObject *func_closure;
    PyObject *func_classobj;
    void *defaults;
    int defaults_pyobjects;
    int flags;
    PyObject *defaults_tuple;
    PyObject *defaults_kwdict;
    PyObject *(*defaults_getter)(PyObject *);
    PyObject *func_annotations;
} __pyx_CyFunctionObject;
static PyTypeObject *__pyx_CyFunctionType = 0;
#define __Pyx_CyFunction_NewEx(ml, flags, qualname, self, module, globals, code)\
    __Pyx_CyFunction_New(__pyx_CyFunctionType, ml, flags, qualname, self, module, globals, code)
static PyObject *__Pyx_CyFunction_New(PyTypeObject *, PyMethodDef *ml,
                                      int flags, PyObject* qualname,
                                      PyObject *self,
                                      PyObject *module, PyObject *globals,
                                      PyObject* code);
static CYTHON_INLINE void *__Pyx_CyFunction_InitDefaults(PyObject *m,
                                                         size_t size,
                                                         int pyobjects);
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsTuple(PyObject *m,
                                                            PyObject *tuple);
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsKwDict(PyObject *m,
                                                             PyObject *dict);
static CYTHON_INLINE void __Pyx_CyFunction_SetAnnotationsDict(PyObject *m,
                                                              PyObject *dict);
static int __pyx_CyFunction_init(void);

/* BufferFallbackError.proto */
static void __Pyx_RaiseBufferFallbackError(void);

/* None.proto */
static CYTHON_INLINE Py_ssize_t __Pyx_div_Py_ssize_t(Py_ssize_t, Py_ssize_t);

/* BufferIndexError.proto */
static void __Pyx_RaiseBufferIndexError(int axis);

#define __Pyx_BufPtrStrided1d(type, buf, i0, s0) (type)((char*)buf + i0 * s0)
/* PySequenceContains.proto */
static CYTHON_INLINE int __Pyx_PySequence_ContainsTF(PyObject* item, PyObject* seq, int eq) {
    int result = PySequence_Contains(seq, item);
    return unlikely(result < 0) ? result : (result == (eq == Py_EQ));
}

/* RaiseTooManyValuesToUnpack.proto */
static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected);

/* RaiseNeedMoreValuesToUnpack.proto */
static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index);

/* RaiseNoneIterError.proto */
static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void);

/* SaveResetException.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_ExceptionSave(type, value, tb)  __Pyx__ExceptionSave(__pyx_tstate, type, value, tb)
static CYTHON_INLINE void __Pyx__ExceptionSave(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb);
#define __Pyx_ExceptionReset(type, value, tb)  __Pyx__ExceptionReset(__pyx_tstate, type, value, tb)
static CYTHON_INLINE void __Pyx__ExceptionReset(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb);
#else
#define __Pyx_ExceptionSave(type, value, tb)   PyErr_GetExcInfo(type, value, tb)
#define __Pyx_ExceptionReset(type, value, tb)  PyErr_SetExcInfo(type, value, tb)
#endif

/* PyErrExceptionMatches.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_PyErr_ExceptionMatches(err) __Pyx_PyErr_ExceptionMatchesInState(__pyx_tstate, err)
static CYTHON_INLINE int __Pyx_PyErr_ExceptionMatchesInState(PyThreadState* tstate, PyObject* err);
#else
#define __Pyx_PyErr_ExceptionMatches(err)  PyErr_ExceptionMatches(err)
#endif

/* GetException.proto */
#if CYTHON_FAST_THREAD_STATE
#define __Pyx_GetException(type, value, tb)  __Pyx__GetException(__pyx_tstate, type, value, tb)
static int __Pyx__GetException(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb);
#else
static int __Pyx_GetException(PyObject **type, PyObject **value, PyObject **tb);
#endif

/* PyObject_GenericGetAttrNoDict.proto */
#if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000
static CYTHON_INLINE PyObject* __Pyx_PyObject_GenericGetAttrNoDict(PyObject* obj, PyObject* attr_name);
#else
#define __Pyx_PyObject_GenericGetAttrNoDict PyObject_GenericGetAttr
#endif

/* PyObject_GenericGetAttr.proto */
#if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000
static PyObject* __Pyx_PyObject_GenericGetAttr(PyObject* obj, PyObject* attr_name);
#else
#define __Pyx_PyObject_GenericGetAttr PyObject_GenericGetAttr
#endif

/* SetupReduce.proto */
static int __Pyx_setup_reduce(PyObject* type_obj);

/* Import.proto */
static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level);

/* CLineInTraceback.proto */
#ifdef CYTHON_CLINE_IN_TRACEBACK
#define __Pyx_CLineForTraceback(tstate, c_line)  (((CYTHON_CLINE_IN_TRACEBACK)) ? c_line : 0)
#else
static int __Pyx_CLineForTraceback(PyThreadState *tstate, int c_line);
#endif

/* CodeObjectCache.proto */
typedef struct {
    PyCodeObject* code_object;
    int code_line;
} __Pyx_CodeObjectCacheEntry;
struct __Pyx_CodeObjectCache {
    int count;
    int max_count;
    __Pyx_CodeObjectCacheEntry* entries;
};
static struct __Pyx_CodeObjectCache __pyx_code_cache = {0,0,NULL};
static int __pyx_bisect_code_objects(__Pyx_CodeObjectCacheEntry* entries, int count, int code_line);
static PyCodeObject *__pyx_find_code_object(int code_line);
static void __pyx_insert_code_object(int code_line, PyCodeObject* code_object);

/* AddTraceback.proto */
static void __Pyx_AddTraceback(const char *funcname, int c_line,
                               int py_line, const char *filename);

/* BufferStructDeclare.proto */
typedef struct {
  Py_ssize_t shape, strides, suboffsets;
} __Pyx_Buf_DimInfo;
typedef struct {
  size_t refcount;
  Py_buffer pybuffer;
} __Pyx_Buffer;
typedef struct {
  __Pyx_Buffer *rcbuffer;
  char *data;
  __Pyx_Buf_DimInfo diminfo[8];
} __Pyx_LocalBuf_ND;

#if PY_MAJOR_VERSION < 3
    static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags);
    static void __Pyx_ReleaseBuffer(Py_buffer *view);
#else
    #define __Pyx_GetBuffer PyObject_GetBuffer
    #define __Pyx_ReleaseBuffer PyBuffer_Release
#endif


/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_long(long value);

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_siz(siz value);

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_Py_intptr_t(Py_intptr_t value);

/* RealImag.proto */
#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #define __Pyx_CREAL(z) ((z).real())
    #define __Pyx_CIMAG(z) ((z).imag())
  #else
    #define __Pyx_CREAL(z) (__real__(z))
    #define __Pyx_CIMAG(z) (__imag__(z))
  #endif
#else
    #define __Pyx_CREAL(z) ((z).real)
    #define __Pyx_CIMAG(z) ((z).imag)
#endif
#if defined(__cplusplus) && CYTHON_CCOMPLEX\
        && (defined(_WIN32) || defined(__clang__) || (defined(__GNUC__) && (__GNUC__ >= 5 || __GNUC__ == 4 && __GNUC_MINOR__ >= 4 )) || __cplusplus >= 201103)
    #define __Pyx_SET_CREAL(z,x) ((z).real(x))
    #define __Pyx_SET_CIMAG(z,y) ((z).imag(y))
#else
    #define __Pyx_SET_CREAL(z,x) __Pyx_CREAL(z) = (x)
    #define __Pyx_SET_CIMAG(z,y) __Pyx_CIMAG(z) = (y)
#endif

/* Arithmetic.proto */
#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq_float(a, b)   ((a)==(b))
    #define __Pyx_c_sum_float(a, b)  ((a)+(b))
    #define __Pyx_c_diff_float(a, b) ((a)-(b))
    #define __Pyx_c_prod_float(a, b) ((a)*(b))
    #define __Pyx_c_quot_float(a, b) ((a)/(b))
    #define __Pyx_c_neg_float(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero_float(z) ((z)==(float)0)
    #define __Pyx_c_conj_float(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_abs_float(z)     (::std::abs(z))
        #define __Pyx_c_pow_float(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zero_float(z) ((z)==0)
    #define __Pyx_c_conj_float(z)    (conjf(z))
    #if 1
        #define __Pyx_c_abs_float(z)     (cabsf(z))
        #define __Pyx_c_pow_float(a, b)  (cpowf(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sum_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_diff_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prod_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quot_float(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_neg_float(__pyx_t_float_complex);
    static CYTHON_INLINE int __Pyx_c_is_zero_float(__pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conj_float(__pyx_t_float_complex);
    #if 1
        static CYTHON_INLINE float __Pyx_c_abs_float(__pyx_t_float_complex);
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_pow_float(__pyx_t_float_complex, __pyx_t_float_complex);
    #endif
#endif

/* Arithmetic.proto */
#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq_double(a, b)   ((a)==(b))
    #define __Pyx_c_sum_double(a, b)  ((a)+(b))
    #define __Pyx_c_diff_double(a, b) ((a)-(b))
    #define __Pyx_c_prod_double(a, b) ((a)*(b))
    #define __Pyx_c_quot_double(a, b) ((a)/(b))
    #define __Pyx_c_neg_double(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero_double(z) ((z)==(double)0)
    #define __Pyx_c_conj_double(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_abs_double(z)     (::std::abs(z))
        #define __Pyx_c_pow_double(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zero_double(z) ((z)==0)
    #define __Pyx_c_conj_double(z)    (conj(z))
    #if 1
        #define __Pyx_c_abs_double(z)     (cabs(z))
        #define __Pyx_c_pow_double(a, b)  (cpow(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot_double(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg_double(__pyx_t_double_complex);
    static CYTHON_INLINE int __Pyx_c_is_zero_double(__pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj_double(__pyx_t_double_complex);
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs_double(__pyx_t_double_complex);
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow_double(__pyx_t_double_complex, __pyx_t_double_complex);
    #endif
#endif

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_int(int value);

/* CIntToPy.proto */
static CYTHON_INLINE PyObject* __Pyx_PyInt_From_enum__NPY_TYPES(enum NPY_TYPES value);

/* CIntFromPy.proto */
static CYTHON_INLINE siz __Pyx_PyInt_As_siz(PyObject *);

/* CIntFromPy.proto */
static CYTHON_INLINE size_t __Pyx_PyInt_As_size_t(PyObject *);

/* CIntFromPy.proto */
static CYTHON_INLINE int __Pyx_PyInt_As_int(PyObject *);

/* CIntFromPy.proto */
static CYTHON_INLINE long __Pyx_PyInt_As_long(PyObject *);

/* FastTypeChecks.proto */
#if CYTHON_COMPILING_IN_CPYTHON
#define __Pyx_TypeCheck(obj, type) __Pyx_IsSubtype(Py_TYPE(obj), (PyTypeObject *)type)
static CYTHON_INLINE int __Pyx_IsSubtype(PyTypeObject *a, PyTypeObject *b);
static CYTHON_INLINE int __Pyx_PyErr_GivenExceptionMatches(PyObject *err, PyObject *type);
static CYTHON_INLINE int __Pyx_PyErr_GivenExceptionMatches2(PyObject *err, PyObject *type1, PyObject *type2);
#else
#define __Pyx_TypeCheck(obj, type) PyObject_TypeCheck(obj, (PyTypeObject *)type)
#define __Pyx_PyErr_GivenExceptionMatches(err, type) PyErr_GivenExceptionMatches(err, type)
#define __Pyx_PyErr_GivenExceptionMatches2(err, type1, type2) (PyErr_GivenExceptionMatches(err, type1) || PyErr_GivenExceptionMatches(err, type2))
#endif
#define __Pyx_PyException_Check(obj) __Pyx_TypeCheck(obj, PyExc_Exception)

/* CheckBinaryVersion.proto */
static int __Pyx_check_binary_version(void);

/* PyIdentifierFromString.proto */
#if !defined(__Pyx_PyIdentifier_FromString)
#if PY_MAJOR_VERSION < 3
  #define __Pyx_PyIdentifier_FromString(s) PyString_FromString(s)
#else
  #define __Pyx_PyIdentifier_FromString(s) PyUnicode_FromString(s)
#endif
#endif

/* ModuleImport.proto */
static PyObject *__Pyx_ImportModule(const char *name);

/* TypeImport.proto */
static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name, size_t size, int strict);

/* InitStrings.proto */
static int __Pyx_InitStrings(__Pyx_StringTabEntry *t);


/* Module declarations from 'cpython.buffer' */

/* Module declarations from 'libc.string' */

/* Module declarations from 'libc.stdio' */

/* Module declarations from '__builtin__' */

/* Module declarations from 'cpython.type' */
static PyTypeObject *__pyx_ptype_7cpython_4type_type = 0;

/* Module declarations from 'cpython' */

/* Module declarations from 'cpython.object' */

/* Module declarations from 'cpython.ref' */

/* Module declarations from 'cpython.mem' */

/* Module declarations from 'numpy' */

/* Module declarations from 'numpy' */
static PyTypeObject *__pyx_ptype_5numpy_dtype = 0;
static PyTypeObject *__pyx_ptype_5numpy_flatiter = 0;
static PyTypeObject *__pyx_ptype_5numpy_broadcast = 0;
static PyTypeObject *__pyx_ptype_5numpy_ndarray = 0;
static PyTypeObject *__pyx_ptype_5numpy_ufunc = 0;
static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *, char *, char *, int *); /*proto*/
static CYTHON_INLINE int __pyx_f_5numpy_import_array(void); /*proto*/

/* Module declarations from 'libc.stdlib' */

/* Module declarations from 'crowdposetools._mask' */
static PyTypeObject *__pyx_ptype_14crowdposetools_5_mask_RLEs = 0;
static PyTypeObject *__pyx_ptype_14crowdposetools_5_mask_Masks = 0;
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_uint8_t = { "uint8_t", NULL, sizeof(__pyx_t_5numpy_uint8_t), { 0 }, 0, IS_UNSIGNED(__pyx_t_5numpy_uint8_t) ? 'U' : 'I', IS_UNSIGNED(__pyx_t_5numpy_uint8_t), 0 };
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_double_t = { "double_t", NULL, sizeof(__pyx_t_5numpy_double_t), { 0 }, 0, 'R', 0, 0 };
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_uint32_t = { "uint32_t", NULL, sizeof(__pyx_t_5numpy_uint32_t), { 0 }, 0, IS_UNSIGNED(__pyx_t_5numpy_uint32_t) ? 'U' : 'I', IS_UNSIGNED(__pyx_t_5numpy_uint32_t), 0 };
#define __Pyx_MODULE_NAME "crowdposetools._mask"
extern int __pyx_module_is_main_crowdposetools___mask;
int __pyx_module_is_main_crowdposetools___mask = 0;

/* Implementation of 'crowdposetools._mask' */
static PyObject *__pyx_builtin_range;
static PyObject *__pyx_builtin_AttributeError;
static PyObject *__pyx_builtin_TypeError;
static PyObject *__pyx_builtin_enumerate;
static PyObject *__pyx_builtin_ValueError;
static PyObject *__pyx_builtin_RuntimeError;
static PyObject *__pyx_builtin_ImportError;
static const char __pyx_k_F[] = "F";
static const char __pyx_k_N[] = "N";
static const char __pyx_k_R[] = "R";
static const char __pyx_k_a[] = "_a";
static const char __pyx_k_h[] = "h";
static const char __pyx_k_i[] = "i";
static const char __pyx_k_j[] = "j";
static const char __pyx_k_m[] = "m";
static const char __pyx_k_n[] = "n";
static const char __pyx_k_p[] = "p";
static const char __pyx_k_w[] = "w";
static const char __pyx_k_Rs[] = "Rs";
static const char __pyx_k_bb[] = "bb";
static const char __pyx_k_dt[] = "dt";
static const char __pyx_k_gt[] = "gt";
static const char __pyx_k_np[] = "np";
static const char __pyx_k_a_2[] = "a";
static const char __pyx_k_all[] = "all";
static const char __pyx_k_iou[] = "_iou";
static const char __pyx_k_len[] = "_len";
static const char __pyx_k_obj[] = "obj";
static const char __pyx_k_sys[] = "sys";
static const char __pyx_k_area[] = "area";
static const char __pyx_k_bb_2[] = "_bb";
static const char __pyx_k_cnts[] = "cnts";
static const char __pyx_k_data[] = "data";
static const char __pyx_k_main[] = "__main__";
static const char __pyx_k_mask[] = "mask";
static const char __pyx_k_name[] = "__name__";
static const char __pyx_k_objs[] = "objs";
static const char __pyx_k_poly[] = "poly";
static const char __pyx_k_size[] = "size";
static const char __pyx_k_test[] = "__test__";
static const char __pyx_k_utf8[] = "utf8";
static const char __pyx_k_array[] = "array";
static const char __pyx_k_bbIou[] = "_bbIou";
static const char __pyx_k_dtype[] = "dtype";
static const char __pyx_k_iou_2[] = "iou";
static const char __pyx_k_isbox[] = "isbox";
static const char __pyx_k_isrle[] = "isrle";
static const char __pyx_k_masks[] = "masks";
static const char __pyx_k_merge[] = "merge";
static const char __pyx_k_numpy[] = "numpy";
static const char __pyx_k_order[] = "order";
static const char __pyx_k_pyobj[] = "pyobj";
static const char __pyx_k_range[] = "range";
static const char __pyx_k_shape[] = "shape";
static const char __pyx_k_uint8[] = "uint8";
static const char __pyx_k_zeros[] = "zeros";
static const char __pyx_k_astype[] = "astype";
static const char __pyx_k_author[] = "__author__";
static const char __pyx_k_counts[] = "counts";
static const char __pyx_k_decode[] = "decode";
static const char __pyx_k_double[] = "double";
static const char __pyx_k_encode[] = "encode";
static const char __pyx_k_frBbox[] = "frBbox";
static const char __pyx_k_frPoly[] = "frPoly";
static const char __pyx_k_import[] = "__import__";
static const char __pyx_k_iouFun[] = "_iouFun";
static const char __pyx_k_reduce[] = "__reduce__";
static const char __pyx_k_rleIou[] = "_rleIou";
static const char __pyx_k_toBbox[] = "toBbox";
static const char __pyx_k_ucRles[] = "ucRles";
static const char __pyx_k_uint32[] = "uint32";
static const char __pyx_k_iscrowd[] = "iscrowd";
static const char __pyx_k_np_poly[] = "np_poly";
static const char __pyx_k_preproc[] = "_preproc";
static const char __pyx_k_reshape[] = "reshape";
static const char __pyx_k_rleObjs[] = "rleObjs";
static const char __pyx_k_tsungyi[] = "tsungyi";
static const char __pyx_k_c_string[] = "c_string";
static const char __pyx_k_frString[] = "_frString";
static const char __pyx_k_getstate[] = "__getstate__";
static const char __pyx_k_setstate[] = "__setstate__";
static const char __pyx_k_toString[] = "_toString";
static const char __pyx_k_TypeError[] = "TypeError";
static const char __pyx_k_enumerate[] = "enumerate";
static const char __pyx_k_intersect[] = "intersect";
static const char __pyx_k_py_string[] = "py_string";
static const char __pyx_k_pyiscrowd[] = "pyiscrowd";
static const char __pyx_k_reduce_ex[] = "__reduce_ex__";
static const char __pyx_k_ValueError[] = "ValueError";
static const char __pyx_k_ImportError[] = "ImportError";
static const char __pyx_k_frPyObjects[] = "frPyObjects";
static const char __pyx_k_RuntimeError[] = "RuntimeError";
static const char __pyx_k_version_info[] = "version_info";
static const char __pyx_k_reduce_cython[] = "__reduce_cython__";
static const char __pyx_k_AttributeError[] = "AttributeError";
static const char __pyx_k_PYTHON_VERSION[] = "PYTHON_VERSION";
static const char __pyx_k_iou_locals__len[] = "iou.<locals>._len";
static const char __pyx_k_setstate_cython[] = "__setstate_cython__";
static const char __pyx_k_frUncompressedRLE[] = "frUncompressedRLE";
static const char __pyx_k_iou_locals__bbIou[] = "iou.<locals>._bbIou";
static const char __pyx_k_cline_in_traceback[] = "cline_in_traceback";
static const char __pyx_k_iou_locals__rleIou[] = "iou.<locals>._rleIou";
static const char __pyx_k_iou_locals__preproc[] = "iou.<locals>._preproc";
static const char __pyx_k_crowdposetools__mask[] = "crowdposetools._mask";
static const char __pyx_k_crowdposetools__mask_pyx[] = "crowdposetools/_mask.pyx";
static const char __pyx_k_input_data_type_not_allowed[] = "input data type not allowed.";
static const char __pyx_k_input_type_is_not_supported[] = "input type is not supported.";
static const char __pyx_k_ndarray_is_not_C_contiguous[] = "ndarray is not C contiguous";
static const char __pyx_k_Python_version_must_be_2_or_3[] = "Python version must be 2 or 3";
static const char __pyx_k_numpy_core_multiarray_failed_to[] = "numpy.core.multiarray failed to import";
static const char __pyx_k_numpy_ndarray_input_is_only_for[] = "numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension";
static const char __pyx_k_unknown_dtype_code_in_numpy_pxd[] = "unknown dtype code in numpy.pxd (%d)";
static const char __pyx_k_unrecognized_type_The_following[] = "unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.";
static const char __pyx_k_Format_string_allocated_too_shor[] = "Format string allocated too short, see comment in numpy.pxd";
static const char __pyx_k_Non_native_byte_order_not_suppor[] = "Non-native byte order not supported";
static const char __pyx_k_The_dt_and_gt_should_have_the_sa[] = "The dt and gt should have the same data type, either RLEs, list or np.ndarray";
static const char __pyx_k_list_input_can_be_bounding_box_N[] = "list input can be bounding box (Nx4) or RLEs ([RLE])";
static const char __pyx_k_ndarray_is_not_Fortran_contiguou[] = "ndarray is not Fortran contiguous";
static const char __pyx_k_no_default___reduce___due_to_non[] = "no default __reduce__ due to non-trivial __cinit__";
static const char __pyx_k_numpy_core_umath_failed_to_impor[] = "numpy.core.umath failed to import";
static const char __pyx_k_Format_string_allocated_too_shor_2[] = "Format string allocated too short.";
static PyObject *__pyx_n_s_AttributeError;
static PyObject *__pyx_n_s_F;
static PyObject *__pyx_kp_u_Format_string_allocated_too_shor;
static PyObject *__pyx_kp_u_Format_string_allocated_too_shor_2;
static PyObject *__pyx_n_s_ImportError;
static PyObject *__pyx_n_s_N;
static PyObject *__pyx_kp_u_Non_native_byte_order_not_suppor;
static PyObject *__pyx_n_s_PYTHON_VERSION;
static PyObject *__pyx_kp_s_Python_version_must_be_2_or_3;
static PyObject *__pyx_n_s_R;
static PyObject *__pyx_n_s_Rs;
static PyObject *__pyx_n_s_RuntimeError;
static PyObject *__pyx_kp_s_The_dt_and_gt_should_have_the_sa;
static PyObject *__pyx_n_s_TypeError;
static PyObject *__pyx_n_s_ValueError;
static PyObject *__pyx_n_s_a;
static PyObject *__pyx_n_s_a_2;
static PyObject *__pyx_n_s_all;
static PyObject *__pyx_n_s_area;
static PyObject *__pyx_n_s_array;
static PyObject *__pyx_n_s_astype;
static PyObject *__pyx_n_s_author;
static PyObject *__pyx_n_s_bb;
static PyObject *__pyx_n_s_bbIou;
static PyObject *__pyx_n_s_bb_2;
static PyObject *__pyx_n_s_c_string;
static PyObject *__pyx_n_s_cline_in_traceback;
static PyObject *__pyx_n_s_cnts;
static PyObject *__pyx_n_s_counts;
static PyObject *__pyx_n_s_crowdposetools__mask;
static PyObject *__pyx_kp_s_crowdposetools__mask_pyx;
static PyObject *__pyx_n_s_data;
static PyObject *__pyx_n_s_decode;
static PyObject *__pyx_n_s_double;
static PyObject *__pyx_n_s_dt;
static PyObject *__pyx_n_s_dtype;
static PyObject *__pyx_n_s_encode;
static PyObject *__pyx_n_s_enumerate;
static PyObject *__pyx_n_s_frBbox;
static PyObject *__pyx_n_s_frPoly;
static PyObject *__pyx_n_s_frPyObjects;
static PyObject *__pyx_n_s_frString;
static PyObject *__pyx_n_s_frUncompressedRLE;
static PyObject *__pyx_n_s_getstate;
static PyObject *__pyx_n_s_gt;
static PyObject *__pyx_n_s_h;
static PyObject *__pyx_n_s_i;
static PyObject *__pyx_n_s_import;
static PyObject *__pyx_kp_s_input_data_type_not_allowed;
static PyObject *__pyx_kp_s_input_type_is_not_supported;
static PyObject *__pyx_n_s_intersect;
static PyObject *__pyx_n_s_iou;
static PyObject *__pyx_n_s_iouFun;
static PyObject *__pyx_n_s_iou_2;
static PyObject *__pyx_n_s_iou_locals__bbIou;
static PyObject *__pyx_n_s_iou_locals__len;
static PyObject *__pyx_n_s_iou_locals__preproc;
static PyObject *__pyx_n_s_iou_locals__rleIou;
static PyObject *__pyx_n_s_isbox;
static PyObject *__pyx_n_s_iscrowd;
static PyObject *__pyx_n_s_isrle;
static PyObject *__pyx_n_s_j;
static PyObject *__pyx_n_s_len;
static PyObject *__pyx_kp_s_list_input_can_be_bounding_box_N;
static PyObject *__pyx_n_s_m;
static PyObject *__pyx_n_s_main;
static PyObject *__pyx_n_s_mask;
static PyObject *__pyx_n_s_masks;
static PyObject *__pyx_n_s_merge;
static PyObject *__pyx_n_s_n;
static PyObject *__pyx_n_s_name;
static PyObject *__pyx_kp_u_ndarray_is_not_C_contiguous;
static PyObject *__pyx_kp_u_ndarray_is_not_Fortran_contiguou;
static PyObject *__pyx_kp_s_no_default___reduce___due_to_non;
static PyObject *__pyx_n_s_np;
static PyObject *__pyx_n_s_np_poly;
static PyObject *__pyx_n_s_numpy;
static PyObject *__pyx_kp_s_numpy_core_multiarray_failed_to;
static PyObject *__pyx_kp_s_numpy_core_umath_failed_to_impor;
static PyObject *__pyx_kp_s_numpy_ndarray_input_is_only_for;
static PyObject *__pyx_n_s_obj;
static PyObject *__pyx_n_s_objs;
static PyObject *__pyx_n_s_order;
static PyObject *__pyx_n_s_p;
static PyObject *__pyx_n_s_poly;
static PyObject *__pyx_n_s_preproc;
static PyObject *__pyx_n_s_py_string;
static PyObject *__pyx_n_s_pyiscrowd;
static PyObject *__pyx_n_s_pyobj;
static PyObject *__pyx_n_s_range;
static PyObject *__pyx_n_s_reduce;
static PyObject *__pyx_n_s_reduce_cython;
static PyObject *__pyx_n_s_reduce_ex;
static PyObject *__pyx_n_s_reshape;
static PyObject *__pyx_n_s_rleIou;
static PyObject *__pyx_n_s_rleObjs;
static PyObject *__pyx_n_s_setstate;
static PyObject *__pyx_n_s_setstate_cython;
static PyObject *__pyx_n_s_shape;
static PyObject *__pyx_n_s_size;
static PyObject *__pyx_n_s_sys;
static PyObject *__pyx_n_s_test;
static PyObject *__pyx_n_s_toBbox;
static PyObject *__pyx_n_s_toString;
static PyObject *__pyx_n_s_tsungyi;
static PyObject *__pyx_n_s_ucRles;
static PyObject *__pyx_n_s_uint32;
static PyObject *__pyx_n_s_uint8;
static PyObject *__pyx_kp_u_unknown_dtype_code_in_numpy_pxd;
static PyObject *__pyx_kp_s_unrecognized_type_The_following;
static PyObject *__pyx_n_s_utf8;
static PyObject *__pyx_n_s_version_info;
static PyObject *__pyx_n_s_w;
static PyObject *__pyx_n_s_zeros;
static int __pyx_pf_14crowdposetools_5_mask_4RLEs___cinit__(struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self, siz __pyx_v_n); /* proto */
static void __pyx_pf_14crowdposetools_5_mask_4RLEs_2__dealloc__(struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_4RLEs_4__getattr__(struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self, PyObject *__pyx_v_key); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_4RLEs_6__reduce_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_4RLEs_8__setstate_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self, CYTHON_UNUSED PyObject *__pyx_v___pyx_state); /* proto */
static int __pyx_pf_14crowdposetools_5_mask_5Masks___cinit__(struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self, PyObject *__pyx_v_h, PyObject *__pyx_v_w, PyObject *__pyx_v_n); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_5Masks_2__array__(struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_5Masks_4__reduce_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_5Masks_6__setstate_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self, CYTHON_UNUSED PyObject *__pyx_v___pyx_state); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask__toString(CYTHON_UNUSED PyObject *__pyx_self, struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_2_frString(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_4encode(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_mask); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_6decode(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_8merge(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs, PyObject *__pyx_v_intersect); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_10area(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou__preproc(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_objs); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou_2_rleIou(CYTHON_UNUSED PyObject *__pyx_self, struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_dt, struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_gt, PyArrayObject *__pyx_v_iscrowd, siz __pyx_v_m, siz __pyx_v_n, PyArrayObject *__pyx_v__iou); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou_4_bbIou(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_dt, PyArrayObject *__pyx_v_gt, PyArrayObject *__pyx_v_iscrowd, siz __pyx_v_m, siz __pyx_v_n, PyArrayObject *__pyx_v__iou); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou_6_len(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_obj); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_12iou(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_dt, PyObject *__pyx_v_gt, PyObject *__pyx_v_pyiscrowd); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_14toBbox(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_16frBbox(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_bb, siz __pyx_v_h, siz __pyx_v_w); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_18frPoly(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_poly, siz __pyx_v_h, siz __pyx_v_w); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_20frUncompressedRLE(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_ucRles, CYTHON_UNUSED siz __pyx_v_h, CYTHON_UNUSED siz __pyx_v_w); /* proto */
static PyObject *__pyx_pf_14crowdposetools_5_mask_22frPyObjects(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_pyobj, PyObject *__pyx_v_h, PyObject *__pyx_v_w); /* proto */
static int __pyx_pf_5numpy_7ndarray___getbuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags); /* proto */
static void __pyx_pf_5numpy_7ndarray_2__releasebuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info); /* proto */
static PyObject *__pyx_tp_new_14crowdposetools_5_mask_RLEs(PyTypeObject *t, PyObject *a, PyObject *k); /*proto*/
static PyObject *__pyx_tp_new_14crowdposetools_5_mask_Masks(PyTypeObject *t, PyObject *a, PyObject *k); /*proto*/
static PyObject *__pyx_int_0;
static PyObject *__pyx_int_1;
static PyObject *__pyx_int_2;
static PyObject *__pyx_int_3;
static PyObject *__pyx_int_4;
static PyObject *__pyx_tuple_;
static PyObject *__pyx_tuple__2;
static PyObject *__pyx_tuple__3;
static PyObject *__pyx_tuple__4;
static PyObject *__pyx_tuple__5;
static PyObject *__pyx_tuple__6;
static PyObject *__pyx_tuple__7;
static PyObject *__pyx_tuple__8;
static PyObject *__pyx_tuple__9;
static PyObject *__pyx_tuple__10;
static PyObject *__pyx_tuple__11;
static PyObject *__pyx_tuple__13;
static PyObject *__pyx_tuple__15;
static PyObject *__pyx_tuple__17;
static PyObject *__pyx_tuple__19;
static PyObject *__pyx_tuple__20;
static PyObject *__pyx_tuple__21;
static PyObject *__pyx_tuple__22;
static PyObject *__pyx_tuple__23;
static PyObject *__pyx_tuple__24;
static PyObject *__pyx_tuple__25;
static PyObject *__pyx_tuple__26;
static PyObject *__pyx_tuple__27;
static PyObject *__pyx_tuple__28;
static PyObject *__pyx_tuple__29;
static PyObject *__pyx_tuple__30;
static PyObject *__pyx_tuple__31;
static PyObject *__pyx_tuple__32;
static PyObject *__pyx_tuple__34;
static PyObject *__pyx_tuple__36;
static PyObject *__pyx_tuple__38;
static PyObject *__pyx_tuple__40;
static PyObject *__pyx_tuple__42;
static PyObject *__pyx_tuple__44;
static PyObject *__pyx_tuple__46;
static PyObject *__pyx_tuple__48;
static PyObject *__pyx_tuple__50;
static PyObject *__pyx_tuple__52;
static PyObject *__pyx_tuple__54;
static PyObject *__pyx_codeobj__12;
static PyObject *__pyx_codeobj__14;
static PyObject *__pyx_codeobj__16;
static PyObject *__pyx_codeobj__18;
static PyObject *__pyx_codeobj__33;
static PyObject *__pyx_codeobj__35;
static PyObject *__pyx_codeobj__37;
static PyObject *__pyx_codeobj__39;
static PyObject *__pyx_codeobj__41;
static PyObject *__pyx_codeobj__43;
static PyObject *__pyx_codeobj__45;
static PyObject *__pyx_codeobj__47;
static PyObject *__pyx_codeobj__49;
static PyObject *__pyx_codeobj__51;
static PyObject *__pyx_codeobj__53;
static PyObject *__pyx_codeobj__55;
/* Late includes */

/* "crowdposetools/_mask.pyx":60
 *     cdef siz _n
 * 
 *     def __cinit__(self, siz n =0):             # <<<<<<<<<<<<<<
 *         rlesInit(&self._R, n)
 *         self._n = n
 */

/* Python wrapper */
static int __pyx_pw_14crowdposetools_5_mask_4RLEs_1__cinit__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static int __pyx_pw_14crowdposetools_5_mask_4RLEs_1__cinit__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  siz __pyx_v_n;
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__cinit__ (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_n,0};
    PyObject* values[1] = {0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (kw_args > 0) {
          PyObject* value = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_n);
          if (value) { values[0] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "__cinit__") < 0)) __PYX_ERR(0, 60, __pyx_L3_error)
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    if (values[0]) {
      __pyx_v_n = __Pyx_PyInt_As_siz(values[0]); if (unlikely((__pyx_v_n == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 60, __pyx_L3_error)
    } else {
      __pyx_v_n = ((siz)0);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("__cinit__", 0, 0, 1, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 60, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.RLEs.__cinit__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return -1;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_4RLEs___cinit__(((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_v_self), __pyx_v_n);

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static int __pyx_pf_14crowdposetools_5_mask_4RLEs___cinit__(struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self, siz __pyx_v_n) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__cinit__", 0);

  /* "crowdposetools/_mask.pyx":61
 * 
 *     def __cinit__(self, siz n =0):
 *         rlesInit(&self._R, n)             # <<<<<<<<<<<<<<
 *         self._n = n
 * 
 */
  rlesInit((&__pyx_v_self->_R), __pyx_v_n);

  /* "crowdposetools/_mask.pyx":62
 *     def __cinit__(self, siz n =0):
 *         rlesInit(&self._R, n)
 *         self._n = n             # <<<<<<<<<<<<<<
 * 
 *     # free the RLE array here
 */
  __pyx_v_self->_n = __pyx_v_n;

  /* "crowdposetools/_mask.pyx":60
 *     cdef siz _n
 * 
 *     def __cinit__(self, siz n =0):             # <<<<<<<<<<<<<<
 *         rlesInit(&self._R, n)
 *         self._n = n
 */

  /* function exit code */
  __pyx_r = 0;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":65
 * 
 *     # free the RLE array here
 *     def __dealloc__(self):             # <<<<<<<<<<<<<<
 *         if self._R is not NULL:
 *             for i in range(self._n):
 */

/* Python wrapper */
static void __pyx_pw_14crowdposetools_5_mask_4RLEs_3__dealloc__(PyObject *__pyx_v_self); /*proto*/
static void __pyx_pw_14crowdposetools_5_mask_4RLEs_3__dealloc__(PyObject *__pyx_v_self) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__dealloc__ (wrapper)", 0);
  __pyx_pf_14crowdposetools_5_mask_4RLEs_2__dealloc__(((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_v_self));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

static void __pyx_pf_14crowdposetools_5_mask_4RLEs_2__dealloc__(struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self) {
  siz __pyx_v_i;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  siz __pyx_t_2;
  siz __pyx_t_3;
  siz __pyx_t_4;
  __Pyx_RefNannySetupContext("__dealloc__", 0);

  /* "crowdposetools/_mask.pyx":66
 *     # free the RLE array here
 *     def __dealloc__(self):
 *         if self._R is not NULL:             # <<<<<<<<<<<<<<
 *             for i in range(self._n):
 *                 free(self._R[i].cnts)
 */
  __pyx_t_1 = ((__pyx_v_self->_R != NULL) != 0);
  if (__pyx_t_1) {

    /* "crowdposetools/_mask.pyx":67
 *     def __dealloc__(self):
 *         if self._R is not NULL:
 *             for i in range(self._n):             # <<<<<<<<<<<<<<
 *                 free(self._R[i].cnts)
 *             free(self._R)
 */
    __pyx_t_2 = __pyx_v_self->_n;
    __pyx_t_3 = __pyx_t_2;
    for (__pyx_t_4 = 0; __pyx_t_4 < __pyx_t_3; __pyx_t_4+=1) {
      __pyx_v_i = __pyx_t_4;

      /* "crowdposetools/_mask.pyx":68
 *         if self._R is not NULL:
 *             for i in range(self._n):
 *                 free(self._R[i].cnts)             # <<<<<<<<<<<<<<
 *             free(self._R)
 *     def __getattr__(self, key):
 */
      free((__pyx_v_self->_R[__pyx_v_i]).cnts);
    }

    /* "crowdposetools/_mask.pyx":69
 *             for i in range(self._n):
 *                 free(self._R[i].cnts)
 *             free(self._R)             # <<<<<<<<<<<<<<
 *     def __getattr__(self, key):
 *         if key == 'n':
 */
    free(__pyx_v_self->_R);

    /* "crowdposetools/_mask.pyx":66
 *     # free the RLE array here
 *     def __dealloc__(self):
 *         if self._R is not NULL:             # <<<<<<<<<<<<<<
 *             for i in range(self._n):
 *                 free(self._R[i].cnts)
 */
  }

  /* "crowdposetools/_mask.pyx":65
 * 
 *     # free the RLE array here
 *     def __dealloc__(self):             # <<<<<<<<<<<<<<
 *         if self._R is not NULL:
 *             for i in range(self._n):
 */

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

/* "crowdposetools/_mask.pyx":70
 *                 free(self._R[i].cnts)
 *             free(self._R)
 *     def __getattr__(self, key):             # <<<<<<<<<<<<<<
 *         if key == 'n':
 *             return self._n
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_4RLEs_5__getattr__(PyObject *__pyx_v_self, PyObject *__pyx_v_key); /*proto*/
static PyObject *__pyx_pw_14crowdposetools_5_mask_4RLEs_5__getattr__(PyObject *__pyx_v_self, PyObject *__pyx_v_key) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__getattr__ (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_4RLEs_4__getattr__(((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_v_self), ((PyObject *)__pyx_v_key));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_4RLEs_4__getattr__(struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self, PyObject *__pyx_v_key) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  PyObject *__pyx_t_2 = NULL;
  __Pyx_RefNannySetupContext("__getattr__", 0);

  /* "crowdposetools/_mask.pyx":71
 *             free(self._R)
 *     def __getattr__(self, key):
 *         if key == 'n':             # <<<<<<<<<<<<<<
 *             return self._n
 *         raise AttributeError(key)
 */
  __pyx_t_1 = (__Pyx_PyString_Equals(__pyx_v_key, __pyx_n_s_n, Py_EQ)); if (unlikely(__pyx_t_1 < 0)) __PYX_ERR(0, 71, __pyx_L1_error)
  if (__pyx_t_1) {

    /* "crowdposetools/_mask.pyx":72
 *     def __getattr__(self, key):
 *         if key == 'n':
 *             return self._n             # <<<<<<<<<<<<<<
 *         raise AttributeError(key)
 * 
 */
    __Pyx_XDECREF(__pyx_r);
    __pyx_t_2 = __Pyx_PyInt_From_siz(__pyx_v_self->_n); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 72, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = 0;
    goto __pyx_L0;

    /* "crowdposetools/_mask.pyx":71
 *             free(self._R)
 *     def __getattr__(self, key):
 *         if key == 'n':             # <<<<<<<<<<<<<<
 *             return self._n
 *         raise AttributeError(key)
 */
  }

  /* "crowdposetools/_mask.pyx":73
 *         if key == 'n':
 *             return self._n
 *         raise AttributeError(key)             # <<<<<<<<<<<<<<
 * 
 * # python class to wrap Mask array in C
 */
  __pyx_t_2 = __Pyx_PyObject_CallOneArg(__pyx_builtin_AttributeError, __pyx_v_key); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 73, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_Raise(__pyx_t_2, 0, 0, 0);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __PYX_ERR(0, 73, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":70
 *                 free(self._R[i].cnts)
 *             free(self._R)
 *     def __getattr__(self, key):             # <<<<<<<<<<<<<<
 *         if key == 'n':
 *             return self._n
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_AddTraceback("crowdposetools._mask.RLEs.__getattr__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "(tree fragment)":1
 * def __reduce_cython__(self):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_4RLEs_7__reduce_cython__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused); /*proto*/
static PyObject *__pyx_pw_14crowdposetools_5_mask_4RLEs_7__reduce_cython__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__reduce_cython__ (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_4RLEs_6__reduce_cython__(((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_v_self));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_4RLEs_6__reduce_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("__reduce_cython__", 0);

  /* "(tree fragment)":2
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */
  __pyx_t_1 = __Pyx_PyObject_Call(__pyx_builtin_TypeError, __pyx_tuple_, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 2, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_Raise(__pyx_t_1, 0, 0, 0);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __PYX_ERR(1, 2, __pyx_L1_error)

  /* "(tree fragment)":1
 * def __reduce_cython__(self):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("crowdposetools._mask.RLEs.__reduce_cython__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "(tree fragment)":3
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_4RLEs_9__setstate_cython__(PyObject *__pyx_v_self, PyObject *__pyx_v___pyx_state); /*proto*/
static PyObject *__pyx_pw_14crowdposetools_5_mask_4RLEs_9__setstate_cython__(PyObject *__pyx_v_self, PyObject *__pyx_v___pyx_state) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__setstate_cython__ (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_4RLEs_8__setstate_cython__(((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_v_self), ((PyObject *)__pyx_v___pyx_state));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_4RLEs_8__setstate_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_self, CYTHON_UNUSED PyObject *__pyx_v___pyx_state) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("__setstate_cython__", 0);

  /* "(tree fragment)":4
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 */
  __pyx_t_1 = __Pyx_PyObject_Call(__pyx_builtin_TypeError, __pyx_tuple__2, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 4, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_Raise(__pyx_t_1, 0, 0, 0);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __PYX_ERR(1, 4, __pyx_L1_error)

  /* "(tree fragment)":3
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("crowdposetools._mask.RLEs.__setstate_cython__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":83
 *     cdef siz _n
 * 
 *     def __cinit__(self, h, w, n):             # <<<<<<<<<<<<<<
 *         self._mask = <byte*> malloc(h*w*n* sizeof(byte))
 *         self._h = h
 */

/* Python wrapper */
static int __pyx_pw_14crowdposetools_5_mask_5Masks_1__cinit__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static int __pyx_pw_14crowdposetools_5_mask_5Masks_1__cinit__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_h = 0;
  PyObject *__pyx_v_w = 0;
  PyObject *__pyx_v_n = 0;
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__cinit__ (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_h,&__pyx_n_s_w,&__pyx_n_s_n,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_h)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_w)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("__cinit__", 1, 3, 3, 1); __PYX_ERR(0, 83, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_n)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("__cinit__", 1, 3, 3, 2); __PYX_ERR(0, 83, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "__cinit__") < 0)) __PYX_ERR(0, 83, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
    }
    __pyx_v_h = values[0];
    __pyx_v_w = values[1];
    __pyx_v_n = values[2];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("__cinit__", 1, 3, 3, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 83, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.Masks.__cinit__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return -1;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_5Masks___cinit__(((struct __pyx_obj_14crowdposetools_5_mask_Masks *)__pyx_v_self), __pyx_v_h, __pyx_v_w, __pyx_v_n);

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static int __pyx_pf_14crowdposetools_5_mask_5Masks___cinit__(struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self, PyObject *__pyx_v_h, PyObject *__pyx_v_w, PyObject *__pyx_v_n) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  size_t __pyx_t_4;
  siz __pyx_t_5;
  __Pyx_RefNannySetupContext("__cinit__", 0);

  /* "crowdposetools/_mask.pyx":84
 * 
 *     def __cinit__(self, h, w, n):
 *         self._mask = <byte*> malloc(h*w*n* sizeof(byte))             # <<<<<<<<<<<<<<
 *         self._h = h
 *         self._w = w
 */
  __pyx_t_1 = PyNumber_Multiply(__pyx_v_h, __pyx_v_w); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 84, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyNumber_Multiply(__pyx_t_1, __pyx_v_n); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 84, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyInt_FromSize_t((sizeof(byte))); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 84, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyNumber_Multiply(__pyx_t_2, __pyx_t_1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 84, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_4 = __Pyx_PyInt_As_size_t(__pyx_t_3); if (unlikely((__pyx_t_4 == (size_t)-1) && PyErr_Occurred())) __PYX_ERR(0, 84, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_v_self->_mask = ((byte *)malloc(__pyx_t_4));

  /* "crowdposetools/_mask.pyx":85
 *     def __cinit__(self, h, w, n):
 *         self._mask = <byte*> malloc(h*w*n* sizeof(byte))
 *         self._h = h             # <<<<<<<<<<<<<<
 *         self._w = w
 *         self._n = n
 */
  __pyx_t_5 = __Pyx_PyInt_As_siz(__pyx_v_h); if (unlikely((__pyx_t_5 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 85, __pyx_L1_error)
  __pyx_v_self->_h = __pyx_t_5;

  /* "crowdposetools/_mask.pyx":86
 *         self._mask = <byte*> malloc(h*w*n* sizeof(byte))
 *         self._h = h
 *         self._w = w             # <<<<<<<<<<<<<<
 *         self._n = n
 *     # def __dealloc__(self):
 */
  __pyx_t_5 = __Pyx_PyInt_As_siz(__pyx_v_w); if (unlikely((__pyx_t_5 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 86, __pyx_L1_error)
  __pyx_v_self->_w = __pyx_t_5;

  /* "crowdposetools/_mask.pyx":87
 *         self._h = h
 *         self._w = w
 *         self._n = n             # <<<<<<<<<<<<<<
 *     # def __dealloc__(self):
 *         # the memory management of _mask has been passed to np.ndarray
 */
  __pyx_t_5 = __Pyx_PyInt_As_siz(__pyx_v_n); if (unlikely((__pyx_t_5 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 87, __pyx_L1_error)
  __pyx_v_self->_n = __pyx_t_5;

  /* "crowdposetools/_mask.pyx":83
 *     cdef siz _n
 * 
 *     def __cinit__(self, h, w, n):             # <<<<<<<<<<<<<<
 *         self._mask = <byte*> malloc(h*w*n* sizeof(byte))
 *         self._h = h
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_AddTraceback("crowdposetools._mask.Masks.__cinit__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":93
 * 
 *     # called when passing into np.array() and return an np.ndarray in column-major order
 *     def __array__(self):             # <<<<<<<<<<<<<<
 *         cdef np.npy_intp shape[1]
 *         shape[0] = <np.npy_intp> self._h*self._w*self._n
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_5Masks_3__array__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused); /*proto*/
static PyObject *__pyx_pw_14crowdposetools_5_mask_5Masks_3__array__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__array__ (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_5Masks_2__array__(((struct __pyx_obj_14crowdposetools_5_mask_Masks *)__pyx_v_self));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_5Masks_2__array__(struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self) {
  npy_intp __pyx_v_shape[1];
  PyObject *__pyx_v_ndarray = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  __Pyx_RefNannySetupContext("__array__", 0);

  /* "crowdposetools/_mask.pyx":95
 *     def __array__(self):
 *         cdef np.npy_intp shape[1]
 *         shape[0] = <np.npy_intp> self._h*self._w*self._n             # <<<<<<<<<<<<<<
 *         # Create a 1D array, and reshape it to fortran/Matlab column-major array
 *         ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._mask).reshape((self._h, self._w, self._n), order='F')
 */
  (__pyx_v_shape[0]) = ((((npy_intp)__pyx_v_self->_h) * __pyx_v_self->_w) * __pyx_v_self->_n);

  /* "crowdposetools/_mask.pyx":97
 *         shape[0] = <np.npy_intp> self._h*self._w*self._n
 *         # Create a 1D array, and reshape it to fortran/Matlab column-major array
 *         ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._mask).reshape((self._h, self._w, self._n), order='F')             # <<<<<<<<<<<<<<
 *         # The _mask allocated by Masks is now handled by ndarray
 *         PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)
 */
  __pyx_t_1 = PyArray_SimpleNewFromData(1, __pyx_v_shape, NPY_UINT8, __pyx_v_self->_mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_reshape); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyInt_From_siz(__pyx_v_self->_h); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = __Pyx_PyInt_From_siz(__pyx_v_self->_w); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_4 = __Pyx_PyInt_From_siz(__pyx_v_self->_n); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_5 = PyTuple_New(3); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_3);
  PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_t_3);
  __Pyx_GIVEREF(__pyx_t_4);
  PyTuple_SET_ITEM(__pyx_t_5, 2, __pyx_t_4);
  __pyx_t_1 = 0;
  __pyx_t_3 = 0;
  __pyx_t_4 = 0;
  __pyx_t_4 = PyTuple_New(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_GIVEREF(__pyx_t_5);
  PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  if (PyDict_SetItem(__pyx_t_5, __pyx_n_s_order, __pyx_n_s_F) < 0) __PYX_ERR(0, 97, __pyx_L1_error)
  __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_4, __pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 97, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_v_ndarray = __pyx_t_3;
  __pyx_t_3 = 0;

  /* "crowdposetools/_mask.pyx":99
 *         ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._mask).reshape((self._h, self._w, self._n), order='F')
 *         # The _mask allocated by Masks is now handled by ndarray
 *         PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)             # <<<<<<<<<<<<<<
 *         return ndarray
 * 
 */
  if (!(likely(((__pyx_v_ndarray) == Py_None) || likely(__Pyx_TypeTest(__pyx_v_ndarray, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 99, __pyx_L1_error)
  PyArray_ENABLEFLAGS(((PyArrayObject *)__pyx_v_ndarray), NPY_OWNDATA);

  /* "crowdposetools/_mask.pyx":100
 *         # The _mask allocated by Masks is now handled by ndarray
 *         PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)
 *         return ndarray             # <<<<<<<<<<<<<<
 * 
 * # internal conversion from Python RLEs object to compressed RLE format
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_ndarray);
  __pyx_r = __pyx_v_ndarray;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":93
 * 
 *     # called when passing into np.array() and return an np.ndarray in column-major order
 *     def __array__(self):             # <<<<<<<<<<<<<<
 *         cdef np.npy_intp shape[1]
 *         shape[0] = <np.npy_intp> self._h*self._w*self._n
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_AddTraceback("crowdposetools._mask.Masks.__array__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF(__pyx_v_ndarray);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "(tree fragment)":1
 * def __reduce_cython__(self):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_5Masks_5__reduce_cython__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused); /*proto*/
static PyObject *__pyx_pw_14crowdposetools_5_mask_5Masks_5__reduce_cython__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__reduce_cython__ (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_5Masks_4__reduce_cython__(((struct __pyx_obj_14crowdposetools_5_mask_Masks *)__pyx_v_self));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_5Masks_4__reduce_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("__reduce_cython__", 0);

  /* "(tree fragment)":2
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */
  __pyx_t_1 = __Pyx_PyObject_Call(__pyx_builtin_TypeError, __pyx_tuple__3, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 2, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_Raise(__pyx_t_1, 0, 0, 0);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __PYX_ERR(1, 2, __pyx_L1_error)

  /* "(tree fragment)":1
 * def __reduce_cython__(self):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("crowdposetools._mask.Masks.__reduce_cython__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "(tree fragment)":3
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_5Masks_7__setstate_cython__(PyObject *__pyx_v_self, PyObject *__pyx_v___pyx_state); /*proto*/
static PyObject *__pyx_pw_14crowdposetools_5_mask_5Masks_7__setstate_cython__(PyObject *__pyx_v_self, PyObject *__pyx_v___pyx_state) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__setstate_cython__ (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_5Masks_6__setstate_cython__(((struct __pyx_obj_14crowdposetools_5_mask_Masks *)__pyx_v_self), ((PyObject *)__pyx_v___pyx_state));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_5Masks_6__setstate_cython__(CYTHON_UNUSED struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_self, CYTHON_UNUSED PyObject *__pyx_v___pyx_state) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("__setstate_cython__", 0);

  /* "(tree fragment)":4
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 */
  __pyx_t_1 = __Pyx_PyObject_Call(__pyx_builtin_TypeError, __pyx_tuple__4, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(1, 4, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_Raise(__pyx_t_1, 0, 0, 0);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __PYX_ERR(1, 4, __pyx_L1_error)

  /* "(tree fragment)":3
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):             # <<<<<<<<<<<<<<
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("crowdposetools._mask.Masks.__setstate_cython__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":103
 * 
 * # internal conversion from Python RLEs object to compressed RLE format
 * def _toString(RLEs Rs):             # <<<<<<<<<<<<<<
 *     cdef siz n = Rs.n
 *     cdef bytes py_string
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_1_toString(PyObject *__pyx_self, PyObject *__pyx_v_Rs); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_1_toString = {"_toString", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_1_toString, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_1_toString(PyObject *__pyx_self, PyObject *__pyx_v_Rs) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_toString (wrapper)", 0);
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_Rs), __pyx_ptype_14crowdposetools_5_mask_RLEs, 1, "Rs", 0))) __PYX_ERR(0, 103, __pyx_L1_error)
  __pyx_r = __pyx_pf_14crowdposetools_5_mask__toString(__pyx_self, ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_v_Rs));

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask__toString(CYTHON_UNUSED PyObject *__pyx_self, struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs) {
  siz __pyx_v_n;
  PyObject *__pyx_v_py_string = 0;
  char *__pyx_v_c_string;
  PyObject *__pyx_v_objs = NULL;
  siz __pyx_v_i;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  siz __pyx_t_2;
  siz __pyx_t_3;
  siz __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  int __pyx_t_8;
  __Pyx_RefNannySetupContext("_toString", 0);

  /* "crowdposetools/_mask.pyx":104
 * # internal conversion from Python RLEs object to compressed RLE format
 * def _toString(RLEs Rs):
 *     cdef siz n = Rs.n             # <<<<<<<<<<<<<<
 *     cdef bytes py_string
 *     cdef char* c_string
 */
  __pyx_t_1 = __Pyx_PyObject_GetAttrStr(((PyObject *)__pyx_v_Rs), __pyx_n_s_n); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 104, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyInt_As_siz(__pyx_t_1); if (unlikely((__pyx_t_2 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 104, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_n = __pyx_t_2;

  /* "crowdposetools/_mask.pyx":107
 *     cdef bytes py_string
 *     cdef char* c_string
 *     objs = []             # <<<<<<<<<<<<<<
 *     for i in range(n):
 *         c_string = rleToString( <RLE*> &Rs._R[i] )
 */
  __pyx_t_1 = PyList_New(0); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 107, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_v_objs = ((PyObject*)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":108
 *     cdef char* c_string
 *     objs = []
 *     for i in range(n):             # <<<<<<<<<<<<<<
 *         c_string = rleToString( <RLE*> &Rs._R[i] )
 *         py_string = c_string
 */
  __pyx_t_2 = __pyx_v_n;
  __pyx_t_3 = __pyx_t_2;
  for (__pyx_t_4 = 0; __pyx_t_4 < __pyx_t_3; __pyx_t_4+=1) {
    __pyx_v_i = __pyx_t_4;

    /* "crowdposetools/_mask.pyx":109
 *     objs = []
 *     for i in range(n):
 *         c_string = rleToString( <RLE*> &Rs._R[i] )             # <<<<<<<<<<<<<<
 *         py_string = c_string
 *         objs.append({
 */
    __pyx_v_c_string = rleToString(((RLE *)(&(__pyx_v_Rs->_R[__pyx_v_i]))));

    /* "crowdposetools/_mask.pyx":110
 *     for i in range(n):
 *         c_string = rleToString( <RLE*> &Rs._R[i] )
 *         py_string = c_string             # <<<<<<<<<<<<<<
 *         objs.append({
 *             'size': [Rs._R[i].h, Rs._R[i].w],
 */
    __pyx_t_1 = __Pyx_PyBytes_FromString(__pyx_v_c_string); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 110, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_XDECREF_SET(__pyx_v_py_string, ((PyObject*)__pyx_t_1));
    __pyx_t_1 = 0;

    /* "crowdposetools/_mask.pyx":112
 *         py_string = c_string
 *         objs.append({
 *             'size': [Rs._R[i].h, Rs._R[i].w],             # <<<<<<<<<<<<<<
 *             'counts': py_string
 *         })
 */
    __pyx_t_1 = __Pyx_PyDict_NewPresized(2); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_5 = __Pyx_PyInt_From_siz((__pyx_v_Rs->_R[__pyx_v_i]).h); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_t_6 = __Pyx_PyInt_From_siz((__pyx_v_Rs->_R[__pyx_v_i]).w); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __pyx_t_7 = PyList_New(2); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __Pyx_GIVEREF(__pyx_t_5);
    PyList_SET_ITEM(__pyx_t_7, 0, __pyx_t_5);
    __Pyx_GIVEREF(__pyx_t_6);
    PyList_SET_ITEM(__pyx_t_7, 1, __pyx_t_6);
    __pyx_t_5 = 0;
    __pyx_t_6 = 0;
    if (PyDict_SetItem(__pyx_t_1, __pyx_n_s_size, __pyx_t_7) < 0) __PYX_ERR(0, 112, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;

    /* "crowdposetools/_mask.pyx":113
 *         objs.append({
 *             'size': [Rs._R[i].h, Rs._R[i].w],
 *             'counts': py_string             # <<<<<<<<<<<<<<
 *         })
 *         free(c_string)
 */
    if (PyDict_SetItem(__pyx_t_1, __pyx_n_s_counts, __pyx_v_py_string) < 0) __PYX_ERR(0, 112, __pyx_L1_error)

    /* "crowdposetools/_mask.pyx":111
 *         c_string = rleToString( <RLE*> &Rs._R[i] )
 *         py_string = c_string
 *         objs.append({             # <<<<<<<<<<<<<<
 *             'size': [Rs._R[i].h, Rs._R[i].w],
 *             'counts': py_string
 */
    __pyx_t_8 = __Pyx_PyList_Append(__pyx_v_objs, __pyx_t_1); if (unlikely(__pyx_t_8 == ((int)-1))) __PYX_ERR(0, 111, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

    /* "crowdposetools/_mask.pyx":115
 *             'counts': py_string
 *         })
 *         free(c_string)             # <<<<<<<<<<<<<<
 *     return objs
 * 
 */
    free(__pyx_v_c_string);
  }

  /* "crowdposetools/_mask.pyx":116
 *         })
 *         free(c_string)
 *     return objs             # <<<<<<<<<<<<<<
 * 
 * # internal conversion from compressed RLE format to Python RLEs object
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_objs);
  __pyx_r = __pyx_v_objs;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":103
 * 
 * # internal conversion from Python RLEs object to compressed RLE format
 * def _toString(RLEs Rs):             # <<<<<<<<<<<<<<
 *     cdef siz n = Rs.n
 *     cdef bytes py_string
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_AddTraceback("crowdposetools._mask._toString", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF(__pyx_v_py_string);
  __Pyx_XDECREF(__pyx_v_objs);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":119
 * 
 * # internal conversion from compressed RLE format to Python RLEs object
 * def _frString(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef siz n = len(rleObjs)
 *     Rs = RLEs(n)
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_3_frString(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_3_frString = {"_frString", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_3_frString, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_3_frString(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_frString (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_2_frString(__pyx_self, ((PyObject *)__pyx_v_rleObjs));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_2_frString(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  siz __pyx_v_n;
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = NULL;
  PyObject *__pyx_v_py_string = 0;
  char *__pyx_v_c_string;
  PyObject *__pyx_v_i = NULL;
  PyObject *__pyx_v_obj = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  Py_ssize_t __pyx_t_1;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *(*__pyx_t_4)(PyObject *);
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  int __pyx_t_7;
  PyObject *__pyx_t_8 = NULL;
  PyObject *__pyx_t_9 = NULL;
  PyObject *__pyx_t_10 = NULL;
  PyObject *__pyx_t_11 = NULL;
  char *__pyx_t_12;
  Py_ssize_t __pyx_t_13;
  siz __pyx_t_14;
  siz __pyx_t_15;
  __Pyx_RefNannySetupContext("_frString", 0);

  /* "crowdposetools/_mask.pyx":120
 * # internal conversion from compressed RLE format to Python RLEs object
 * def _frString(rleObjs):
 *     cdef siz n = len(rleObjs)             # <<<<<<<<<<<<<<
 *     Rs = RLEs(n)
 *     cdef bytes py_string
 */
  __pyx_t_1 = PyObject_Length(__pyx_v_rleObjs); if (unlikely(__pyx_t_1 == ((Py_ssize_t)-1))) __PYX_ERR(0, 120, __pyx_L1_error)
  __pyx_v_n = __pyx_t_1;

  /* "crowdposetools/_mask.pyx":121
 * def _frString(rleObjs):
 *     cdef siz n = len(rleObjs)
 *     Rs = RLEs(n)             # <<<<<<<<<<<<<<
 *     cdef bytes py_string
 *     cdef char* c_string
 */
  __pyx_t_2 = __Pyx_PyInt_From_siz(__pyx_v_n); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 121, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_3 = __Pyx_PyObject_CallOneArg(((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), __pyx_t_2); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 121, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_3);
  __pyx_t_3 = 0;

  /* "crowdposetools/_mask.pyx":124
 *     cdef bytes py_string
 *     cdef char* c_string
 *     for i, obj in enumerate(rleObjs):             # <<<<<<<<<<<<<<
 *         if PYTHON_VERSION == 2:
 *             py_string = str(obj['counts']).encode('utf8')
 */
  __Pyx_INCREF(__pyx_int_0);
  __pyx_t_3 = __pyx_int_0;
  if (likely(PyList_CheckExact(__pyx_v_rleObjs)) || PyTuple_CheckExact(__pyx_v_rleObjs)) {
    __pyx_t_2 = __pyx_v_rleObjs; __Pyx_INCREF(__pyx_t_2); __pyx_t_1 = 0;
    __pyx_t_4 = NULL;
  } else {
    __pyx_t_1 = -1; __pyx_t_2 = PyObject_GetIter(__pyx_v_rleObjs); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 124, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __pyx_t_4 = Py_TYPE(__pyx_t_2)->tp_iternext; if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 124, __pyx_L1_error)
  }
  for (;;) {
    if (likely(!__pyx_t_4)) {
      if (likely(PyList_CheckExact(__pyx_t_2))) {
        if (__pyx_t_1 >= PyList_GET_SIZE(__pyx_t_2)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = PyList_GET_ITEM(__pyx_t_2, __pyx_t_1); __Pyx_INCREF(__pyx_t_5); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 124, __pyx_L1_error)
        #else
        __pyx_t_5 = PySequence_ITEM(__pyx_t_2, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 124, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        #endif
      } else {
        if (__pyx_t_1 >= PyTuple_GET_SIZE(__pyx_t_2)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = PyTuple_GET_ITEM(__pyx_t_2, __pyx_t_1); __Pyx_INCREF(__pyx_t_5); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 124, __pyx_L1_error)
        #else
        __pyx_t_5 = PySequence_ITEM(__pyx_t_2, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 124, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        #endif
      }
    } else {
      __pyx_t_5 = __pyx_t_4(__pyx_t_2);
      if (unlikely(!__pyx_t_5)) {
        PyObject* exc_type = PyErr_Occurred();
        if (exc_type) {
          if (likely(__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();
          else __PYX_ERR(0, 124, __pyx_L1_error)
        }
        break;
      }
      __Pyx_GOTREF(__pyx_t_5);
    }
    __Pyx_XDECREF_SET(__pyx_v_obj, __pyx_t_5);
    __pyx_t_5 = 0;
    __Pyx_INCREF(__pyx_t_3);
    __Pyx_XDECREF_SET(__pyx_v_i, __pyx_t_3);
    __pyx_t_5 = __Pyx_PyInt_AddObjC(__pyx_t_3, __pyx_int_1, 1, 0); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 124, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_3);
    __pyx_t_3 = __pyx_t_5;
    __pyx_t_5 = 0;

    /* "crowdposetools/_mask.pyx":125
 *     cdef char* c_string
 *     for i, obj in enumerate(rleObjs):
 *         if PYTHON_VERSION == 2:             # <<<<<<<<<<<<<<
 *             py_string = str(obj['counts']).encode('utf8')
 *         elif PYTHON_VERSION == 3:
 */
    __pyx_t_5 = __Pyx_GetModuleGlobalName(__pyx_n_s_PYTHON_VERSION); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 125, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_t_6 = __Pyx_PyInt_EqObjC(__pyx_t_5, __pyx_int_2, 2, 0); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 125, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_6); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 125, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    if (__pyx_t_7) {

      /* "crowdposetools/_mask.pyx":126
 *     for i, obj in enumerate(rleObjs):
 *         if PYTHON_VERSION == 2:
 *             py_string = str(obj['counts']).encode('utf8')             # <<<<<<<<<<<<<<
 *         elif PYTHON_VERSION == 3:
 *             py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
 */
      __pyx_t_6 = __Pyx_PyObject_Dict_GetItem(__pyx_v_obj, __pyx_n_s_counts); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 126, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __pyx_t_5 = __Pyx_PyObject_CallOneArg(((PyObject *)(&PyString_Type)), __pyx_t_6); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 126, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      __pyx_t_6 = __Pyx_PyObject_GetAttrStr(__pyx_t_5, __pyx_n_s_encode); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 126, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_5 = __Pyx_PyObject_Call(__pyx_t_6, __pyx_tuple__5, NULL); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 126, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      if (!(likely(PyBytes_CheckExact(__pyx_t_5))||((__pyx_t_5) == Py_None)||(PyErr_Format(PyExc_TypeError, "Expected %.16s, got %.200s", "bytes", Py_TYPE(__pyx_t_5)->tp_name), 0))) __PYX_ERR(0, 126, __pyx_L1_error)
      __Pyx_XDECREF_SET(__pyx_v_py_string, ((PyObject*)__pyx_t_5));
      __pyx_t_5 = 0;

      /* "crowdposetools/_mask.pyx":125
 *     cdef char* c_string
 *     for i, obj in enumerate(rleObjs):
 *         if PYTHON_VERSION == 2:             # <<<<<<<<<<<<<<
 *             py_string = str(obj['counts']).encode('utf8')
 *         elif PYTHON_VERSION == 3:
 */
      goto __pyx_L5;
    }

    /* "crowdposetools/_mask.pyx":127
 *         if PYTHON_VERSION == 2:
 *             py_string = str(obj['counts']).encode('utf8')
 *         elif PYTHON_VERSION == 3:             # <<<<<<<<<<<<<<
 *             py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
 *         else:
 */
    __pyx_t_5 = __Pyx_GetModuleGlobalName(__pyx_n_s_PYTHON_VERSION); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 127, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_t_6 = __Pyx_PyInt_EqObjC(__pyx_t_5, __pyx_int_3, 3, 0); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 127, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_6); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 127, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    if (likely(__pyx_t_7)) {

      /* "crowdposetools/_mask.pyx":128
 *             py_string = str(obj['counts']).encode('utf8')
 *         elif PYTHON_VERSION == 3:
 *             py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']             # <<<<<<<<<<<<<<
 *         else:
 *             raise Exception('Python version must be 2 or 3')
 */
      __pyx_t_5 = __Pyx_PyObject_Dict_GetItem(__pyx_v_obj, __pyx_n_s_counts); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 128, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_8 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_t_5)), ((PyObject *)(&PyString_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_8); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 128, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_8); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 128, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      if (__pyx_t_7) {
        __pyx_t_5 = __Pyx_PyObject_GetAttrStr(((PyObject *)(&PyString_Type)), __pyx_n_s_encode); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 128, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        __pyx_t_9 = __Pyx_PyObject_Dict_GetItem(__pyx_v_obj, __pyx_n_s_counts); if (unlikely(!__pyx_t_9)) __PYX_ERR(0, 128, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_9);
        __pyx_t_10 = NULL;
        if (CYTHON_UNPACK_METHODS && likely(PyMethod_Check(__pyx_t_5))) {
          __pyx_t_10 = PyMethod_GET_SELF(__pyx_t_5);
          if (likely(__pyx_t_10)) {
            PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_5);
            __Pyx_INCREF(__pyx_t_10);
            __Pyx_INCREF(function);
            __Pyx_DECREF_SET(__pyx_t_5, function);
          }
        }
        if (!__pyx_t_10) {
          __pyx_t_8 = __Pyx_PyObject_CallOneArg(__pyx_t_5, __pyx_t_9); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 128, __pyx_L1_error)
          __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
          __Pyx_GOTREF(__pyx_t_8);
        } else {
          #if CYTHON_FAST_PYCALL
          if (PyFunction_Check(__pyx_t_5)) {
            PyObject *__pyx_temp[2] = {__pyx_t_10, __pyx_t_9};
            __pyx_t_8 = __Pyx_PyFunction_FastCall(__pyx_t_5, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 128, __pyx_L1_error)
            __Pyx_XDECREF(__pyx_t_10); __pyx_t_10 = 0;
            __Pyx_GOTREF(__pyx_t_8);
            __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
          } else
          #endif
          #if CYTHON_FAST_PYCCALL
          if (__Pyx_PyFastCFunction_Check(__pyx_t_5)) {
            PyObject *__pyx_temp[2] = {__pyx_t_10, __pyx_t_9};
            __pyx_t_8 = __Pyx_PyCFunction_FastCall(__pyx_t_5, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 128, __pyx_L1_error)
            __Pyx_XDECREF(__pyx_t_10); __pyx_t_10 = 0;
            __Pyx_GOTREF(__pyx_t_8);
            __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
          } else
          #endif
          {
            __pyx_t_11 = PyTuple_New(1+1); if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 128, __pyx_L1_error)
            __Pyx_GOTREF(__pyx_t_11);
            __Pyx_GIVEREF(__pyx_t_10); PyTuple_SET_ITEM(__pyx_t_11, 0, __pyx_t_10); __pyx_t_10 = NULL;
            __Pyx_GIVEREF(__pyx_t_9);
            PyTuple_SET_ITEM(__pyx_t_11, 0+1, __pyx_t_9);
            __pyx_t_9 = 0;
            __pyx_t_8 = __Pyx_PyObject_Call(__pyx_t_5, __pyx_t_11, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 128, __pyx_L1_error)
            __Pyx_GOTREF(__pyx_t_8);
            __Pyx_DECREF(__pyx_t_11); __pyx_t_11 = 0;
          }
        }
        __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
        if (!(likely(PyBytes_CheckExact(__pyx_t_8))||((__pyx_t_8) == Py_None)||(PyErr_Format(PyExc_TypeError, "Expected %.16s, got %.200s", "bytes", Py_TYPE(__pyx_t_8)->tp_name), 0))) __PYX_ERR(0, 128, __pyx_L1_error)
        __pyx_t_6 = __pyx_t_8;
        __pyx_t_8 = 0;
      } else {
        __pyx_t_8 = __Pyx_PyObject_Dict_GetItem(__pyx_v_obj, __pyx_n_s_counts); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 128, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_8);
        if (!(likely(PyBytes_CheckExact(__pyx_t_8))||((__pyx_t_8) == Py_None)||(PyErr_Format(PyExc_TypeError, "Expected %.16s, got %.200s", "bytes", Py_TYPE(__pyx_t_8)->tp_name), 0))) __PYX_ERR(0, 128, __pyx_L1_error)
        __pyx_t_6 = __pyx_t_8;
        __pyx_t_8 = 0;
      }
      __Pyx_XDECREF_SET(__pyx_v_py_string, ((PyObject*)__pyx_t_6));
      __pyx_t_6 = 0;

      /* "crowdposetools/_mask.pyx":127
 *         if PYTHON_VERSION == 2:
 *             py_string = str(obj['counts']).encode('utf8')
 *         elif PYTHON_VERSION == 3:             # <<<<<<<<<<<<<<
 *             py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
 *         else:
 */
      goto __pyx_L5;
    }

    /* "crowdposetools/_mask.pyx":130
 *             py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
 *         else:
 *             raise Exception('Python version must be 2 or 3')             # <<<<<<<<<<<<<<
 *         c_string = py_string
 *         rleFrString( <RLE*> &Rs._R[i], <char*> c_string, obj['size'][0], obj['size'][1] )
 */
    /*else*/ {
      __pyx_t_6 = __Pyx_PyObject_Call(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])), __pyx_tuple__6, NULL); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 130, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_Raise(__pyx_t_6, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      __PYX_ERR(0, 130, __pyx_L1_error)
    }
    __pyx_L5:;

    /* "crowdposetools/_mask.pyx":131
 *         else:
 *             raise Exception('Python version must be 2 or 3')
 *         c_string = py_string             # <<<<<<<<<<<<<<
 *         rleFrString( <RLE*> &Rs._R[i], <char*> c_string, obj['size'][0], obj['size'][1] )
 *     return Rs
 */
    if (unlikely(__pyx_v_py_string == Py_None)) {
      PyErr_SetString(PyExc_TypeError, "expected bytes, NoneType found");
      __PYX_ERR(0, 131, __pyx_L1_error)
    }
    __pyx_t_12 = __Pyx_PyBytes_AsWritableString(__pyx_v_py_string); if (unlikely((!__pyx_t_12) && PyErr_Occurred())) __PYX_ERR(0, 131, __pyx_L1_error)
    __pyx_v_c_string = __pyx_t_12;

    /* "crowdposetools/_mask.pyx":132
 *             raise Exception('Python version must be 2 or 3')
 *         c_string = py_string
 *         rleFrString( <RLE*> &Rs._R[i], <char*> c_string, obj['size'][0], obj['size'][1] )             # <<<<<<<<<<<<<<
 *     return Rs
 * 
 */
    __pyx_t_13 = __Pyx_PyIndex_AsSsize_t(__pyx_v_i); if (unlikely((__pyx_t_13 == (Py_ssize_t)-1) && PyErr_Occurred())) __PYX_ERR(0, 132, __pyx_L1_error)
    __pyx_t_6 = __Pyx_PyObject_Dict_GetItem(__pyx_v_obj, __pyx_n_s_size); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 132, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __pyx_t_8 = __Pyx_GetItemInt(__pyx_t_6, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 132, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_14 = __Pyx_PyInt_As_siz(__pyx_t_8); if (unlikely((__pyx_t_14 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 132, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_8 = __Pyx_PyObject_Dict_GetItem(__pyx_v_obj, __pyx_n_s_size); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 132, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __pyx_t_6 = __Pyx_GetItemInt(__pyx_t_8, 1, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 132, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_15 = __Pyx_PyInt_As_siz(__pyx_t_6); if (unlikely((__pyx_t_15 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 132, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    rleFrString(((RLE *)(&(__pyx_v_Rs->_R[__pyx_t_13]))), ((char *)__pyx_v_c_string), __pyx_t_14, __pyx_t_15);

    /* "crowdposetools/_mask.pyx":124
 *     cdef bytes py_string
 *     cdef char* c_string
 *     for i, obj in enumerate(rleObjs):             # <<<<<<<<<<<<<<
 *         if PYTHON_VERSION == 2:
 *             py_string = str(obj['counts']).encode('utf8')
 */
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;

  /* "crowdposetools/_mask.pyx":133
 *         c_string = py_string
 *         rleFrString( <RLE*> &Rs._R[i], <char*> c_string, obj['size'][0], obj['size'][1] )
 *     return Rs             # <<<<<<<<<<<<<<
 * 
 * # encode mask to RLEs objects
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(((PyObject *)__pyx_v_Rs));
  __pyx_r = ((PyObject *)__pyx_v_Rs);
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":119
 * 
 * # internal conversion from compressed RLE format to Python RLEs object
 * def _frString(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef siz n = len(rleObjs)
 *     Rs = RLEs(n)
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_XDECREF(__pyx_t_9);
  __Pyx_XDECREF(__pyx_t_10);
  __Pyx_XDECREF(__pyx_t_11);
  __Pyx_AddTraceback("crowdposetools._mask._frString", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF(__pyx_v_py_string);
  __Pyx_XDECREF(__pyx_v_i);
  __Pyx_XDECREF(__pyx_v_obj);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":137
 * # encode mask to RLEs objects
 * # list of RLE string can be generated by RLEs member function
 * def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):             # <<<<<<<<<<<<<<
 *     h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
 *     cdef RLEs Rs = RLEs(n)
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_5encode(PyObject *__pyx_self, PyObject *__pyx_v_mask); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_5encode = {"encode", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_5encode, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_5encode(PyObject *__pyx_self, PyObject *__pyx_v_mask) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("encode (wrapper)", 0);
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_mask), __pyx_ptype_5numpy_ndarray, 1, "mask", 0))) __PYX_ERR(0, 137, __pyx_L1_error)
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_4encode(__pyx_self, ((PyArrayObject *)__pyx_v_mask));

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_4encode(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_mask) {
  npy_intp __pyx_v_h;
  npy_intp __pyx_v_w;
  npy_intp __pyx_v_n;
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = 0;
  PyObject *__pyx_v_objs = NULL;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_mask;
  __Pyx_Buffer __pyx_pybuffer_mask;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  npy_intp __pyx_t_1;
  npy_intp __pyx_t_2;
  npy_intp __pyx_t_3;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  __Pyx_RefNannySetupContext("encode", 0);
  __pyx_pybuffer_mask.pybuffer.buf = NULL;
  __pyx_pybuffer_mask.refcount = 0;
  __pyx_pybuffernd_mask.data = NULL;
  __pyx_pybuffernd_mask.rcbuffer = &__pyx_pybuffer_mask;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_mask.rcbuffer->pybuffer, (PyObject*)__pyx_v_mask, &__Pyx_TypeInfo_nn___pyx_t_5numpy_uint8_t, PyBUF_FORMAT| PyBUF_F_CONTIGUOUS, 3, 0, __pyx_stack) == -1)) __PYX_ERR(0, 137, __pyx_L1_error)
  }
  __pyx_pybuffernd_mask.diminfo[0].strides = __pyx_pybuffernd_mask.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_mask.diminfo[0].shape = __pyx_pybuffernd_mask.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_mask.diminfo[1].strides = __pyx_pybuffernd_mask.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_mask.diminfo[1].shape = __pyx_pybuffernd_mask.rcbuffer->pybuffer.shape[1]; __pyx_pybuffernd_mask.diminfo[2].strides = __pyx_pybuffernd_mask.rcbuffer->pybuffer.strides[2]; __pyx_pybuffernd_mask.diminfo[2].shape = __pyx_pybuffernd_mask.rcbuffer->pybuffer.shape[2];

  /* "crowdposetools/_mask.pyx":138
 * # list of RLE string can be generated by RLEs member function
 * def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):
 *     h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = RLEs(n)
 *     rleEncode(Rs._R,<byte*>mask.data,h,w,n)
 */
  __pyx_t_1 = (__pyx_v_mask->dimensions[0]);
  __pyx_t_2 = (__pyx_v_mask->dimensions[1]);
  __pyx_t_3 = (__pyx_v_mask->dimensions[2]);
  __pyx_v_h = __pyx_t_1;
  __pyx_v_w = __pyx_t_2;
  __pyx_v_n = __pyx_t_3;

  /* "crowdposetools/_mask.pyx":139
 * def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):
 *     h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
 *     cdef RLEs Rs = RLEs(n)             # <<<<<<<<<<<<<<
 *     rleEncode(Rs._R,<byte*>mask.data,h,w,n)
 *     objs = _toString(Rs)
 */
  __pyx_t_4 = __Pyx_PyInt_From_Py_intptr_t(__pyx_v_n); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 139, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_5 = __Pyx_PyObject_CallOneArg(((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), __pyx_t_4); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 139, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_5);
  __pyx_t_5 = 0;

  /* "crowdposetools/_mask.pyx":140
 *     h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
 *     cdef RLEs Rs = RLEs(n)
 *     rleEncode(Rs._R,<byte*>mask.data,h,w,n)             # <<<<<<<<<<<<<<
 *     objs = _toString(Rs)
 *     return objs
 */
  rleEncode(__pyx_v_Rs->_R, ((byte *)__pyx_v_mask->data), __pyx_v_h, __pyx_v_w, __pyx_v_n);

  /* "crowdposetools/_mask.pyx":141
 *     cdef RLEs Rs = RLEs(n)
 *     rleEncode(Rs._R,<byte*>mask.data,h,w,n)
 *     objs = _toString(Rs)             # <<<<<<<<<<<<<<
 *     return objs
 * 
 */
  __pyx_t_4 = __Pyx_GetModuleGlobalName(__pyx_n_s_toString); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 141, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_6 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_4))) {
    __pyx_t_6 = PyMethod_GET_SELF(__pyx_t_4);
    if (likely(__pyx_t_6)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_4);
      __Pyx_INCREF(__pyx_t_6);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_4, function);
    }
  }
  if (!__pyx_t_6) {
    __pyx_t_5 = __Pyx_PyObject_CallOneArg(__pyx_t_4, ((PyObject *)__pyx_v_Rs)); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 141, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_4)) {
      PyObject *__pyx_temp[2] = {__pyx_t_6, ((PyObject *)__pyx_v_Rs)};
      __pyx_t_5 = __Pyx_PyFunction_FastCall(__pyx_t_4, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 141, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_5);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_4)) {
      PyObject *__pyx_temp[2] = {__pyx_t_6, ((PyObject *)__pyx_v_Rs)};
      __pyx_t_5 = __Pyx_PyCFunction_FastCall(__pyx_t_4, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 141, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_5);
    } else
    #endif
    {
      __pyx_t_7 = PyTuple_New(1+1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 141, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_7);
      __Pyx_GIVEREF(__pyx_t_6); PyTuple_SET_ITEM(__pyx_t_7, 0, __pyx_t_6); __pyx_t_6 = NULL;
      __Pyx_INCREF(((PyObject *)__pyx_v_Rs));
      __Pyx_GIVEREF(((PyObject *)__pyx_v_Rs));
      PyTuple_SET_ITEM(__pyx_t_7, 0+1, ((PyObject *)__pyx_v_Rs));
      __pyx_t_5 = __Pyx_PyObject_Call(__pyx_t_4, __pyx_t_7, NULL); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 141, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_v_objs = __pyx_t_5;
  __pyx_t_5 = 0;

  /* "crowdposetools/_mask.pyx":142
 *     rleEncode(Rs._R,<byte*>mask.data,h,w,n)
 *     objs = _toString(Rs)
 *     return objs             # <<<<<<<<<<<<<<
 * 
 * # decode mask from compressed list of RLE string or RLEs object
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_objs);
  __pyx_r = __pyx_v_objs;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":137
 * # encode mask to RLEs objects
 * # list of RLE string can be generated by RLEs member function
 * def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):             # <<<<<<<<<<<<<<
 *     h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
 *     cdef RLEs Rs = RLEs(n)
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_mask.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("crowdposetools._mask.encode", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_mask.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF(__pyx_v_objs);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":145
 * 
 * # decode mask from compressed list of RLE string or RLEs object
 * def decode(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_7decode(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_7decode = {"decode", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_7decode, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_7decode(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("decode (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_6decode(__pyx_self, ((PyObject *)__pyx_v_rleObjs));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_6decode(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = 0;
  siz __pyx_v_h;
  siz __pyx_v_w;
  siz __pyx_v_n;
  struct __pyx_obj_14crowdposetools_5_mask_Masks *__pyx_v_masks = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  siz __pyx_t_5;
  siz __pyx_t_6;
  siz __pyx_t_7;
  __Pyx_RefNannySetupContext("decode", 0);

  /* "crowdposetools/_mask.pyx":146
 * # decode mask from compressed list of RLE string or RLEs object
 * def decode(rleObjs):
 *     cdef RLEs Rs = _frString(rleObjs)             # <<<<<<<<<<<<<<
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
 *     masks = Masks(h, w, n)
 */
  __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_frString); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 146, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_3 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_2))) {
    __pyx_t_3 = PyMethod_GET_SELF(__pyx_t_2);
    if (likely(__pyx_t_3)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_2);
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_2, function);
    }
  }
  if (!__pyx_t_3) {
    __pyx_t_1 = __Pyx_PyObject_CallOneArg(__pyx_t_2, __pyx_v_rleObjs); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 146, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 146, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 146, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 146, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_GIVEREF(__pyx_t_3); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_3); __pyx_t_3 = NULL;
      __Pyx_INCREF(__pyx_v_rleObjs);
      __Pyx_GIVEREF(__pyx_v_rleObjs);
      PyTuple_SET_ITEM(__pyx_t_4, 0+1, __pyx_v_rleObjs);
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_4, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 146, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_14crowdposetools_5_mask_RLEs))))) __PYX_ERR(0, 146, __pyx_L1_error)
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":147
 * def decode(rleObjs):
 *     cdef RLEs Rs = _frString(rleObjs)
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n             # <<<<<<<<<<<<<<
 *     masks = Masks(h, w, n)
 *     rleDecode(<RLE*>Rs._R, masks._mask, n);
 */
  __pyx_t_5 = (__pyx_v_Rs->_R[0]).h;
  __pyx_t_6 = (__pyx_v_Rs->_R[0]).w;
  __pyx_t_7 = __pyx_v_Rs->_n;
  __pyx_v_h = __pyx_t_5;
  __pyx_v_w = __pyx_t_6;
  __pyx_v_n = __pyx_t_7;

  /* "crowdposetools/_mask.pyx":148
 *     cdef RLEs Rs = _frString(rleObjs)
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
 *     masks = Masks(h, w, n)             # <<<<<<<<<<<<<<
 *     rleDecode(<RLE*>Rs._R, masks._mask, n);
 *     return np.array(masks)
 */
  __pyx_t_1 = __Pyx_PyInt_From_siz(__pyx_v_h); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 148, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyInt_From_siz(__pyx_v_w); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 148, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_4 = __Pyx_PyInt_From_siz(__pyx_v_n); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 148, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_3 = PyTuple_New(3); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 148, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_2);
  PyTuple_SET_ITEM(__pyx_t_3, 1, __pyx_t_2);
  __Pyx_GIVEREF(__pyx_t_4);
  PyTuple_SET_ITEM(__pyx_t_3, 2, __pyx_t_4);
  __pyx_t_1 = 0;
  __pyx_t_2 = 0;
  __pyx_t_4 = 0;
  __pyx_t_4 = __Pyx_PyObject_Call(((PyObject *)__pyx_ptype_14crowdposetools_5_mask_Masks), __pyx_t_3, NULL); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 148, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_v_masks = ((struct __pyx_obj_14crowdposetools_5_mask_Masks *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "crowdposetools/_mask.pyx":149
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
 *     masks = Masks(h, w, n)
 *     rleDecode(<RLE*>Rs._R, masks._mask, n);             # <<<<<<<<<<<<<<
 *     return np.array(masks)
 * 
 */
  rleDecode(((RLE *)__pyx_v_Rs->_R), __pyx_v_masks->_mask, __pyx_v_n);

  /* "crowdposetools/_mask.pyx":150
 *     masks = Masks(h, w, n)
 *     rleDecode(<RLE*>Rs._R, masks._mask, n);
 *     return np.array(masks)             # <<<<<<<<<<<<<<
 * 
 * def merge(rleObjs, intersect=0):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 150, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_t_3, __pyx_n_s_array); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 150, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_3 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_2))) {
    __pyx_t_3 = PyMethod_GET_SELF(__pyx_t_2);
    if (likely(__pyx_t_3)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_2);
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_2, function);
    }
  }
  if (!__pyx_t_3) {
    __pyx_t_4 = __Pyx_PyObject_CallOneArg(__pyx_t_2, ((PyObject *)__pyx_v_masks)); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 150, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, ((PyObject *)__pyx_v_masks)};
      __pyx_t_4 = __Pyx_PyFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 150, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_4);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, ((PyObject *)__pyx_v_masks)};
      __pyx_t_4 = __Pyx_PyCFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 150, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_4);
    } else
    #endif
    {
      __pyx_t_1 = PyTuple_New(1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 150, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_GIVEREF(__pyx_t_3); PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_t_3); __pyx_t_3 = NULL;
      __Pyx_INCREF(((PyObject *)__pyx_v_masks));
      __Pyx_GIVEREF(((PyObject *)__pyx_v_masks));
      PyTuple_SET_ITEM(__pyx_t_1, 0+1, ((PyObject *)__pyx_v_masks));
      __pyx_t_4 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_1, NULL); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 150, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_r = __pyx_t_4;
  __pyx_t_4 = 0;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":145
 * 
 * # decode mask from compressed list of RLE string or RLEs object
 * def decode(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_AddTraceback("crowdposetools._mask.decode", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF((PyObject *)__pyx_v_masks);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":152
 *     return np.array(masks)
 * 
 * def merge(rleObjs, intersect=0):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef RLEs R = RLEs(1)
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_9merge(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_9merge = {"merge", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_9merge, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_9merge(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_rleObjs = 0;
  PyObject *__pyx_v_intersect = 0;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("merge (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_rleObjs,&__pyx_n_s_intersect,0};
    PyObject* values[2] = {0,0};
    values[1] = ((PyObject *)__pyx_int_0);
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_rleObjs)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (kw_args > 0) {
          PyObject* value = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_intersect);
          if (value) { values[1] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "merge") < 0)) __PYX_ERR(0, 152, __pyx_L3_error)
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_rleObjs = values[0];
    __pyx_v_intersect = values[1];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("merge", 0, 1, 2, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 152, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.merge", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_8merge(__pyx_self, __pyx_v_rleObjs, __pyx_v_intersect);

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_8merge(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs, PyObject *__pyx_v_intersect) {
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = 0;
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_R = 0;
  PyObject *__pyx_v_obj = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  int __pyx_t_5;
  __Pyx_RefNannySetupContext("merge", 0);

  /* "crowdposetools/_mask.pyx":153
 * 
 * def merge(rleObjs, intersect=0):
 *     cdef RLEs Rs = _frString(rleObjs)             # <<<<<<<<<<<<<<
 *     cdef RLEs R = RLEs(1)
 *     rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)
 */
  __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_frString); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 153, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_3 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_2))) {
    __pyx_t_3 = PyMethod_GET_SELF(__pyx_t_2);
    if (likely(__pyx_t_3)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_2);
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_2, function);
    }
  }
  if (!__pyx_t_3) {
    __pyx_t_1 = __Pyx_PyObject_CallOneArg(__pyx_t_2, __pyx_v_rleObjs); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 153, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 153, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 153, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 153, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_GIVEREF(__pyx_t_3); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_3); __pyx_t_3 = NULL;
      __Pyx_INCREF(__pyx_v_rleObjs);
      __Pyx_GIVEREF(__pyx_v_rleObjs);
      PyTuple_SET_ITEM(__pyx_t_4, 0+1, __pyx_v_rleObjs);
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_4, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 153, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_14crowdposetools_5_mask_RLEs))))) __PYX_ERR(0, 153, __pyx_L1_error)
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":154
 * def merge(rleObjs, intersect=0):
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef RLEs R = RLEs(1)             # <<<<<<<<<<<<<<
 *     rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)
 *     obj = _toString(R)[0]
 */
  __pyx_t_1 = __Pyx_PyObject_Call(((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), __pyx_tuple__7, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 154, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_v_R = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":155
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef RLEs R = RLEs(1)
 *     rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)             # <<<<<<<<<<<<<<
 *     obj = _toString(R)[0]
 *     return obj
 */
  __pyx_t_5 = __Pyx_PyInt_As_int(__pyx_v_intersect); if (unlikely((__pyx_t_5 == (int)-1) && PyErr_Occurred())) __PYX_ERR(0, 155, __pyx_L1_error)
  rleMerge(((RLE *)__pyx_v_Rs->_R), ((RLE *)__pyx_v_R->_R), ((siz)__pyx_v_Rs->_n), __pyx_t_5);

  /* "crowdposetools/_mask.pyx":156
 *     cdef RLEs R = RLEs(1)
 *     rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)
 *     obj = _toString(R)[0]             # <<<<<<<<<<<<<<
 *     return obj
 * 
 */
  __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_toString); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 156, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_4 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_2))) {
    __pyx_t_4 = PyMethod_GET_SELF(__pyx_t_2);
    if (likely(__pyx_t_4)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_2);
      __Pyx_INCREF(__pyx_t_4);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_2, function);
    }
  }
  if (!__pyx_t_4) {
    __pyx_t_1 = __Pyx_PyObject_CallOneArg(__pyx_t_2, ((PyObject *)__pyx_v_R)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 156, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_4, ((PyObject *)__pyx_v_R)};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 156, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_4, ((PyObject *)__pyx_v_R)};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 156, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_3 = PyTuple_New(1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 156, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_GIVEREF(__pyx_t_4); PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_4); __pyx_t_4 = NULL;
      __Pyx_INCREF(((PyObject *)__pyx_v_R));
      __Pyx_GIVEREF(((PyObject *)__pyx_v_R));
      PyTuple_SET_ITEM(__pyx_t_3, 0+1, ((PyObject *)__pyx_v_R));
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_3, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 156, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = __Pyx_GetItemInt(__pyx_t_1, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 156, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_obj = __pyx_t_2;
  __pyx_t_2 = 0;

  /* "crowdposetools/_mask.pyx":157
 *     rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)
 *     obj = _toString(R)[0]
 *     return obj             # <<<<<<<<<<<<<<
 * 
 * def area(rleObjs):
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_obj);
  __pyx_r = __pyx_v_obj;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":152
 *     return np.array(masks)
 * 
 * def merge(rleObjs, intersect=0):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef RLEs R = RLEs(1)
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_AddTraceback("crowdposetools._mask.merge", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF((PyObject *)__pyx_v_R);
  __Pyx_XDECREF(__pyx_v_obj);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":159
 *     return obj
 * 
 * def area(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_11area(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_11area = {"area", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_11area, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_11area(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("area (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_10area(__pyx_self, ((PyObject *)__pyx_v_rleObjs));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_10area(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = 0;
  uint *__pyx_v__a;
  npy_intp __pyx_v_shape[1];
  PyObject *__pyx_v_a = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  __Pyx_RefNannySetupContext("area", 0);

  /* "crowdposetools/_mask.pyx":160
 * 
 * def area(rleObjs):
 *     cdef RLEs Rs = _frString(rleObjs)             # <<<<<<<<<<<<<<
 *     cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))
 *     rleArea(Rs._R, Rs._n, _a)
 */
  __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_frString); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 160, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_3 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_2))) {
    __pyx_t_3 = PyMethod_GET_SELF(__pyx_t_2);
    if (likely(__pyx_t_3)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_2);
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_2, function);
    }
  }
  if (!__pyx_t_3) {
    __pyx_t_1 = __Pyx_PyObject_CallOneArg(__pyx_t_2, __pyx_v_rleObjs); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 160, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 160, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 160, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 160, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_GIVEREF(__pyx_t_3); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_3); __pyx_t_3 = NULL;
      __Pyx_INCREF(__pyx_v_rleObjs);
      __Pyx_GIVEREF(__pyx_v_rleObjs);
      PyTuple_SET_ITEM(__pyx_t_4, 0+1, __pyx_v_rleObjs);
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_4, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 160, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_14crowdposetools_5_mask_RLEs))))) __PYX_ERR(0, 160, __pyx_L1_error)
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":161
 * def area(rleObjs):
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))             # <<<<<<<<<<<<<<
 *     rleArea(Rs._R, Rs._n, _a)
 *     cdef np.npy_intp shape[1]
 */
  __pyx_v__a = ((uint *)malloc((__pyx_v_Rs->_n * (sizeof(unsigned int)))));

  /* "crowdposetools/_mask.pyx":162
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))
 *     rleArea(Rs._R, Rs._n, _a)             # <<<<<<<<<<<<<<
 *     cdef np.npy_intp shape[1]
 *     shape[0] = <np.npy_intp> Rs._n
 */
  rleArea(__pyx_v_Rs->_R, __pyx_v_Rs->_n, __pyx_v__a);

  /* "crowdposetools/_mask.pyx":164
 *     rleArea(Rs._R, Rs._n, _a)
 *     cdef np.npy_intp shape[1]
 *     shape[0] = <np.npy_intp> Rs._n             # <<<<<<<<<<<<<<
 *     a = np.array((Rs._n, ), dtype=np.uint8)
 *     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
 */
  (__pyx_v_shape[0]) = ((npy_intp)__pyx_v_Rs->_n);

  /* "crowdposetools/_mask.pyx":165
 *     cdef np.npy_intp shape[1]
 *     shape[0] = <np.npy_intp> Rs._n
 *     a = np.array((Rs._n, ), dtype=np.uint8)             # <<<<<<<<<<<<<<
 *     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
 *     PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
 */
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_array); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyInt_From_siz(__pyx_v_Rs->_n); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_4 = PyTuple_New(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyTuple_New(1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_4);
  PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_t_4);
  __pyx_t_4 = 0;
  __pyx_t_4 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_5 = __Pyx_PyObject_GetAttrStr(__pyx_t_3, __pyx_n_s_uint8); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (PyDict_SetItem(__pyx_t_4, __pyx_n_s_dtype, __pyx_t_5) < 0) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_1, __pyx_t_4); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 165, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_v_a = __pyx_t_5;
  __pyx_t_5 = 0;

  /* "crowdposetools/_mask.pyx":166
 *     shape[0] = <np.npy_intp> Rs._n
 *     a = np.array((Rs._n, ), dtype=np.uint8)
 *     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)             # <<<<<<<<<<<<<<
 *     PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
 *     return a
 */
  __pyx_t_5 = PyArray_SimpleNewFromData(1, __pyx_v_shape, NPY_UINT32, __pyx_v__a); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 166, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF_SET(__pyx_v_a, __pyx_t_5);
  __pyx_t_5 = 0;

  /* "crowdposetools/_mask.pyx":167
 *     a = np.array((Rs._n, ), dtype=np.uint8)
 *     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
 *     PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)             # <<<<<<<<<<<<<<
 *     return a
 * 
 */
  if (!(likely(((__pyx_v_a) == Py_None) || likely(__Pyx_TypeTest(__pyx_v_a, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 167, __pyx_L1_error)
  PyArray_ENABLEFLAGS(((PyArrayObject *)__pyx_v_a), NPY_OWNDATA);

  /* "crowdposetools/_mask.pyx":168
 *     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
 *     PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
 *     return a             # <<<<<<<<<<<<<<
 * 
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_a);
  __pyx_r = __pyx_v_a;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":159
 *     return obj
 * 
 * def area(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_AddTraceback("crowdposetools._mask.area", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF(__pyx_v_a);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":171
 * 
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):             # <<<<<<<<<<<<<<
 *     def _preproc(objs):
 *         if len(objs) == 0:
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_13iou(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_13iou = {"iou", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_13iou, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_13iou(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_dt = 0;
  PyObject *__pyx_v_gt = 0;
  PyObject *__pyx_v_pyiscrowd = 0;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("iou (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_dt,&__pyx_n_s_gt,&__pyx_n_s_pyiscrowd,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_dt)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_gt)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("iou", 1, 3, 3, 1); __PYX_ERR(0, 171, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_pyiscrowd)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("iou", 1, 3, 3, 2); __PYX_ERR(0, 171, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "iou") < 0)) __PYX_ERR(0, 171, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
    }
    __pyx_v_dt = values[0];
    __pyx_v_gt = values[1];
    __pyx_v_pyiscrowd = values[2];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("iou", 1, 3, 3, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 171, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.iou", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_12iou(__pyx_self, __pyx_v_dt, __pyx_v_gt, __pyx_v_pyiscrowd);

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":172
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):
 *     def _preproc(objs):             # <<<<<<<<<<<<<<
 *         if len(objs) == 0:
 *             return objs
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_1_preproc(PyObject *__pyx_self, PyObject *__pyx_v_objs); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_3iou_1_preproc = {"_preproc", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_3iou_1_preproc, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_1_preproc(PyObject *__pyx_self, PyObject *__pyx_v_objs) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_preproc (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_3iou__preproc(__pyx_self, ((PyObject *)__pyx_v_objs));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou__preproc(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_objs) {
  PyObject *__pyx_v_isbox = NULL;
  PyObject *__pyx_v_isrle = NULL;
  PyObject *__pyx_v_obj = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  Py_ssize_t __pyx_t_1;
  int __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  int __pyx_t_8;
  int __pyx_t_9;
  PyObject *__pyx_t_10 = NULL;
  PyObject *(*__pyx_t_11)(PyObject *);
  PyObject *__pyx_t_12 = NULL;
  Py_ssize_t __pyx_t_13;
  PyObject *__pyx_t_14 = NULL;
  __Pyx_RefNannySetupContext("_preproc", 0);
  __Pyx_INCREF(__pyx_v_objs);

  /* "crowdposetools/_mask.pyx":173
 * def iou( dt, gt, pyiscrowd ):
 *     def _preproc(objs):
 *         if len(objs) == 0:             # <<<<<<<<<<<<<<
 *             return objs
 *         if type(objs) == np.ndarray:
 */
  __pyx_t_1 = PyObject_Length(__pyx_v_objs); if (unlikely(__pyx_t_1 == ((Py_ssize_t)-1))) __PYX_ERR(0, 173, __pyx_L1_error)
  __pyx_t_2 = ((__pyx_t_1 == 0) != 0);
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":174
 *     def _preproc(objs):
 *         if len(objs) == 0:
 *             return objs             # <<<<<<<<<<<<<<
 *         if type(objs) == np.ndarray:
 *             if len(objs.shape) == 1:
 */
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(__pyx_v_objs);
    __pyx_r = __pyx_v_objs;
    goto __pyx_L0;

    /* "crowdposetools/_mask.pyx":173
 * def iou( dt, gt, pyiscrowd ):
 *     def _preproc(objs):
 *         if len(objs) == 0:             # <<<<<<<<<<<<<<
 *             return objs
 *         if type(objs) == np.ndarray:
 */
  }

  /* "crowdposetools/_mask.pyx":175
 *         if len(objs) == 0:
 *             return objs
 *         if type(objs) == np.ndarray:             # <<<<<<<<<<<<<<
 *             if len(objs.shape) == 1:
 *                 objs = objs.reshape((objs[0], 1))
 */
  __pyx_t_3 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_objs)), ((PyObject *)__pyx_ptype_5numpy_ndarray), Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 175, __pyx_L1_error)
  __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 175, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":176
 *             return objs
 *         if type(objs) == np.ndarray:
 *             if len(objs.shape) == 1:             # <<<<<<<<<<<<<<
 *                 objs = objs.reshape((objs[0], 1))
 *             # check if it's Nx4 bbox
 */
    __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_shape); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 176, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_1 = PyObject_Length(__pyx_t_3); if (unlikely(__pyx_t_1 == ((Py_ssize_t)-1))) __PYX_ERR(0, 176, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_2 = ((__pyx_t_1 == 1) != 0);
    if (__pyx_t_2) {

      /* "crowdposetools/_mask.pyx":177
 *         if type(objs) == np.ndarray:
 *             if len(objs.shape) == 1:
 *                 objs = objs.reshape((objs[0], 1))             # <<<<<<<<<<<<<<
 *             # check if it's Nx4 bbox
 *             if not len(objs.shape) == 2 or not objs.shape[1] == 4:
 */
      __pyx_t_4 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_reshape); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 177, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_5 = __Pyx_GetItemInt(__pyx_v_objs, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 177, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_6 = PyTuple_New(2); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 177, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_GIVEREF(__pyx_t_5);
      PyTuple_SET_ITEM(__pyx_t_6, 0, __pyx_t_5);
      __Pyx_INCREF(__pyx_int_1);
      __Pyx_GIVEREF(__pyx_int_1);
      PyTuple_SET_ITEM(__pyx_t_6, 1, __pyx_int_1);
      __pyx_t_5 = 0;
      __pyx_t_5 = NULL;
      if (CYTHON_UNPACK_METHODS && likely(PyMethod_Check(__pyx_t_4))) {
        __pyx_t_5 = PyMethod_GET_SELF(__pyx_t_4);
        if (likely(__pyx_t_5)) {
          PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_4);
          __Pyx_INCREF(__pyx_t_5);
          __Pyx_INCREF(function);
          __Pyx_DECREF_SET(__pyx_t_4, function);
        }
      }
      if (!__pyx_t_5) {
        __pyx_t_3 = __Pyx_PyObject_CallOneArg(__pyx_t_4, __pyx_t_6); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 177, __pyx_L1_error)
        __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
        __Pyx_GOTREF(__pyx_t_3);
      } else {
        #if CYTHON_FAST_PYCALL
        if (PyFunction_Check(__pyx_t_4)) {
          PyObject *__pyx_temp[2] = {__pyx_t_5, __pyx_t_6};
          __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_4, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 177, __pyx_L1_error)
          __Pyx_XDECREF(__pyx_t_5); __pyx_t_5 = 0;
          __Pyx_GOTREF(__pyx_t_3);
          __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
        } else
        #endif
        #if CYTHON_FAST_PYCCALL
        if (__Pyx_PyFastCFunction_Check(__pyx_t_4)) {
          PyObject *__pyx_temp[2] = {__pyx_t_5, __pyx_t_6};
          __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_4, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 177, __pyx_L1_error)
          __Pyx_XDECREF(__pyx_t_5); __pyx_t_5 = 0;
          __Pyx_GOTREF(__pyx_t_3);
          __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
        } else
        #endif
        {
          __pyx_t_7 = PyTuple_New(1+1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 177, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_7);
          __Pyx_GIVEREF(__pyx_t_5); PyTuple_SET_ITEM(__pyx_t_7, 0, __pyx_t_5); __pyx_t_5 = NULL;
          __Pyx_GIVEREF(__pyx_t_6);
          PyTuple_SET_ITEM(__pyx_t_7, 0+1, __pyx_t_6);
          __pyx_t_6 = 0;
          __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_4, __pyx_t_7, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 177, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_3);
          __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
        }
      }
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_DECREF_SET(__pyx_v_objs, __pyx_t_3);
      __pyx_t_3 = 0;

      /* "crowdposetools/_mask.pyx":176
 *             return objs
 *         if type(objs) == np.ndarray:
 *             if len(objs.shape) == 1:             # <<<<<<<<<<<<<<
 *                 objs = objs.reshape((objs[0], 1))
 *             # check if it's Nx4 bbox
 */
    }

    /* "crowdposetools/_mask.pyx":179
 *                 objs = objs.reshape((objs[0], 1))
 *             # check if it's Nx4 bbox
 *             if not len(objs.shape) == 2 or not objs.shape[1] == 4:             # <<<<<<<<<<<<<<
 *                 raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
 *             objs = objs.astype(np.double)
 */
    __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_shape); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 179, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_1 = PyObject_Length(__pyx_t_3); if (unlikely(__pyx_t_1 == ((Py_ssize_t)-1))) __PYX_ERR(0, 179, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_8 = ((!((__pyx_t_1 == 2) != 0)) != 0);
    if (!__pyx_t_8) {
    } else {
      __pyx_t_2 = __pyx_t_8;
      goto __pyx_L7_bool_binop_done;
    }
    __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_shape); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 179, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_4 = __Pyx_GetItemInt(__pyx_t_3, 1, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 179, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_3 = __Pyx_PyInt_EqObjC(__pyx_t_4, __pyx_int_4, 4, 0); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 179, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __pyx_t_8 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_8 < 0)) __PYX_ERR(0, 179, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_9 = ((!__pyx_t_8) != 0);
    __pyx_t_2 = __pyx_t_9;
    __pyx_L7_bool_binop_done:;
    if (unlikely(__pyx_t_2)) {

      /* "crowdposetools/_mask.pyx":180
 *             # check if it's Nx4 bbox
 *             if not len(objs.shape) == 2 or not objs.shape[1] == 4:
 *                 raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')             # <<<<<<<<<<<<<<
 *             objs = objs.astype(np.double)
 *         elif type(objs) == list:
 */
      __pyx_t_3 = __Pyx_PyObject_Call(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])), __pyx_tuple__8, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 180, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(0, 180, __pyx_L1_error)

      /* "crowdposetools/_mask.pyx":179
 *                 objs = objs.reshape((objs[0], 1))
 *             # check if it's Nx4 bbox
 *             if not len(objs.shape) == 2 or not objs.shape[1] == 4:             # <<<<<<<<<<<<<<
 *                 raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
 *             objs = objs.astype(np.double)
 */
    }

    /* "crowdposetools/_mask.pyx":181
 *             if not len(objs.shape) == 2 or not objs.shape[1] == 4:
 *                 raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
 *             objs = objs.astype(np.double)             # <<<<<<<<<<<<<<
 *         elif type(objs) == list:
 *             # check if list is in box format and convert it to np.ndarray
 */
    __pyx_t_4 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_astype); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 181, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_7 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 181, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __pyx_t_6 = __Pyx_PyObject_GetAttrStr(__pyx_t_7, __pyx_n_s_double); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 181, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    __pyx_t_7 = NULL;
    if (CYTHON_UNPACK_METHODS && likely(PyMethod_Check(__pyx_t_4))) {
      __pyx_t_7 = PyMethod_GET_SELF(__pyx_t_4);
      if (likely(__pyx_t_7)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_4);
        __Pyx_INCREF(__pyx_t_7);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_4, function);
      }
    }
    if (!__pyx_t_7) {
      __pyx_t_3 = __Pyx_PyObject_CallOneArg(__pyx_t_4, __pyx_t_6); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 181, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_3);
    } else {
      #if CYTHON_FAST_PYCALL
      if (PyFunction_Check(__pyx_t_4)) {
        PyObject *__pyx_temp[2] = {__pyx_t_7, __pyx_t_6};
        __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_4, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 181, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_7); __pyx_t_7 = 0;
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      } else
      #endif
      #if CYTHON_FAST_PYCCALL
      if (__Pyx_PyFastCFunction_Check(__pyx_t_4)) {
        PyObject *__pyx_temp[2] = {__pyx_t_7, __pyx_t_6};
        __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_4, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 181, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_7); __pyx_t_7 = 0;
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      } else
      #endif
      {
        __pyx_t_5 = PyTuple_New(1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 181, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        __Pyx_GIVEREF(__pyx_t_7); PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_7); __pyx_t_7 = NULL;
        __Pyx_GIVEREF(__pyx_t_6);
        PyTuple_SET_ITEM(__pyx_t_5, 0+1, __pyx_t_6);
        __pyx_t_6 = 0;
        __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_4, __pyx_t_5, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 181, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      }
    }
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __Pyx_DECREF_SET(__pyx_v_objs, __pyx_t_3);
    __pyx_t_3 = 0;

    /* "crowdposetools/_mask.pyx":175
 *         if len(objs) == 0:
 *             return objs
 *         if type(objs) == np.ndarray:             # <<<<<<<<<<<<<<
 *             if len(objs.shape) == 1:
 *                 objs = objs.reshape((objs[0], 1))
 */
    goto __pyx_L4;
  }

  /* "crowdposetools/_mask.pyx":182
 *                 raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
 *             objs = objs.astype(np.double)
 *         elif type(objs) == list:             # <<<<<<<<<<<<<<
 *             # check if list is in box format and convert it to np.ndarray
 *             isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
 */
  __pyx_t_3 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_objs)), ((PyObject *)(&PyList_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 182, __pyx_L1_error)
  __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 182, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (likely(__pyx_t_2)) {

    /* "crowdposetools/_mask.pyx":184
 *         elif type(objs) == list:
 *             # check if list is in box format and convert it to np.ndarray
 *             isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))             # <<<<<<<<<<<<<<
 *             isrle = np.all(np.array([type(obj) == dict for obj in objs]))
 *             if isbox:
 */
    __pyx_t_4 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 184, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_5 = __Pyx_PyObject_GetAttrStr(__pyx_t_4, __pyx_n_s_all); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 184, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __pyx_t_6 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 184, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __pyx_t_7 = __Pyx_PyObject_GetAttrStr(__pyx_t_6, __pyx_n_s_array); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 184, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_6 = PyList_New(0); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 184, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    if (likely(PyList_CheckExact(__pyx_v_objs)) || PyTuple_CheckExact(__pyx_v_objs)) {
      __pyx_t_10 = __pyx_v_objs; __Pyx_INCREF(__pyx_t_10); __pyx_t_1 = 0;
      __pyx_t_11 = NULL;
    } else {
      __pyx_t_1 = -1; __pyx_t_10 = PyObject_GetIter(__pyx_v_objs); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 184, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_10);
      __pyx_t_11 = Py_TYPE(__pyx_t_10)->tp_iternext; if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 184, __pyx_L1_error)
    }
    for (;;) {
      if (likely(!__pyx_t_11)) {
        if (likely(PyList_CheckExact(__pyx_t_10))) {
          if (__pyx_t_1 >= PyList_GET_SIZE(__pyx_t_10)) break;
          #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
          __pyx_t_12 = PyList_GET_ITEM(__pyx_t_10, __pyx_t_1); __Pyx_INCREF(__pyx_t_12); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 184, __pyx_L1_error)
          #else
          __pyx_t_12 = PySequence_ITEM(__pyx_t_10, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 184, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_12);
          #endif
        } else {
          if (__pyx_t_1 >= PyTuple_GET_SIZE(__pyx_t_10)) break;
          #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
          __pyx_t_12 = PyTuple_GET_ITEM(__pyx_t_10, __pyx_t_1); __Pyx_INCREF(__pyx_t_12); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 184, __pyx_L1_error)
          #else
          __pyx_t_12 = PySequence_ITEM(__pyx_t_10, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 184, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_12);
          #endif
        }
      } else {
        __pyx_t_12 = __pyx_t_11(__pyx_t_10);
        if (unlikely(!__pyx_t_12)) {
          PyObject* exc_type = PyErr_Occurred();
          if (exc_type) {
            if (likely(__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();
            else __PYX_ERR(0, 184, __pyx_L1_error)
          }
          break;
        }
        __Pyx_GOTREF(__pyx_t_12);
      }
      __Pyx_XDECREF_SET(__pyx_v_obj, __pyx_t_12);
      __pyx_t_12 = 0;
      __pyx_t_13 = PyObject_Length(__pyx_v_obj); if (unlikely(__pyx_t_13 == ((Py_ssize_t)-1))) __PYX_ERR(0, 184, __pyx_L1_error)
      __pyx_t_2 = (__pyx_t_13 == 4);
      if (__pyx_t_2) {
      } else {
        __pyx_t_14 = __Pyx_PyBool_FromLong(__pyx_t_2); if (unlikely(!__pyx_t_14)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_14);
        __pyx_t_12 = __pyx_t_14;
        __pyx_t_14 = 0;
        goto __pyx_L11_bool_binop_done;
      }
      __pyx_t_14 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_obj)), ((PyObject *)(&PyList_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_14); if (unlikely(!__pyx_t_14)) __PYX_ERR(0, 184, __pyx_L1_error)
      __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_t_14); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 184, __pyx_L1_error)
      if (!__pyx_t_2) {
        __Pyx_DECREF(__pyx_t_14); __pyx_t_14 = 0;
      } else {
        __Pyx_INCREF(__pyx_t_14);
        __pyx_t_12 = __pyx_t_14;
        __Pyx_DECREF(__pyx_t_14); __pyx_t_14 = 0;
        goto __pyx_L11_bool_binop_done;
      }
      __pyx_t_14 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_obj)), ((PyObject *)__pyx_ptype_5numpy_ndarray), Py_EQ); __Pyx_XGOTREF(__pyx_t_14); if (unlikely(!__pyx_t_14)) __PYX_ERR(0, 184, __pyx_L1_error)
      __Pyx_INCREF(__pyx_t_14);
      __pyx_t_12 = __pyx_t_14;
      __Pyx_DECREF(__pyx_t_14); __pyx_t_14 = 0;
      __pyx_L11_bool_binop_done:;
      if (unlikely(__Pyx_ListComp_Append(__pyx_t_6, (PyObject*)__pyx_t_12))) __PYX_ERR(0, 184, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;
    }
    __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
    __pyx_t_10 = NULL;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_7))) {
      __pyx_t_10 = PyMethod_GET_SELF(__pyx_t_7);
      if (likely(__pyx_t_10)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_7);
        __Pyx_INCREF(__pyx_t_10);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_7, function);
      }
    }
    if (!__pyx_t_10) {
      __pyx_t_4 = __Pyx_PyObject_CallOneArg(__pyx_t_7, __pyx_t_6); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 184, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_4);
    } else {
      #if CYTHON_FAST_PYCALL
      if (PyFunction_Check(__pyx_t_7)) {
        PyObject *__pyx_temp[2] = {__pyx_t_10, __pyx_t_6};
        __pyx_t_4 = __Pyx_PyFunction_FastCall(__pyx_t_7, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_10); __pyx_t_10 = 0;
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      } else
      #endif
      #if CYTHON_FAST_PYCCALL
      if (__Pyx_PyFastCFunction_Check(__pyx_t_7)) {
        PyObject *__pyx_temp[2] = {__pyx_t_10, __pyx_t_6};
        __pyx_t_4 = __Pyx_PyCFunction_FastCall(__pyx_t_7, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_10); __pyx_t_10 = 0;
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
      } else
      #endif
      {
        __pyx_t_12 = PyTuple_New(1+1); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_12);
        __Pyx_GIVEREF(__pyx_t_10); PyTuple_SET_ITEM(__pyx_t_12, 0, __pyx_t_10); __pyx_t_10 = NULL;
        __Pyx_GIVEREF(__pyx_t_6);
        PyTuple_SET_ITEM(__pyx_t_12, 0+1, __pyx_t_6);
        __pyx_t_6 = 0;
        __pyx_t_4 = __Pyx_PyObject_Call(__pyx_t_7, __pyx_t_12, NULL); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;
      }
    }
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    __pyx_t_7 = NULL;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_5))) {
      __pyx_t_7 = PyMethod_GET_SELF(__pyx_t_5);
      if (likely(__pyx_t_7)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_5);
        __Pyx_INCREF(__pyx_t_7);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_5, function);
      }
    }
    if (!__pyx_t_7) {
      __pyx_t_3 = __Pyx_PyObject_CallOneArg(__pyx_t_5, __pyx_t_4); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 184, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_3);
    } else {
      #if CYTHON_FAST_PYCALL
      if (PyFunction_Check(__pyx_t_5)) {
        PyObject *__pyx_temp[2] = {__pyx_t_7, __pyx_t_4};
        __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_5, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_7); __pyx_t_7 = 0;
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      } else
      #endif
      #if CYTHON_FAST_PYCCALL
      if (__Pyx_PyFastCFunction_Check(__pyx_t_5)) {
        PyObject *__pyx_temp[2] = {__pyx_t_7, __pyx_t_4};
        __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_5, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_7); __pyx_t_7 = 0;
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      } else
      #endif
      {
        __pyx_t_12 = PyTuple_New(1+1); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_12);
        __Pyx_GIVEREF(__pyx_t_7); PyTuple_SET_ITEM(__pyx_t_12, 0, __pyx_t_7); __pyx_t_7 = NULL;
        __Pyx_GIVEREF(__pyx_t_4);
        PyTuple_SET_ITEM(__pyx_t_12, 0+1, __pyx_t_4);
        __pyx_t_4 = 0;
        __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_5, __pyx_t_12, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 184, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;
      }
    }
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_isbox = __pyx_t_3;
    __pyx_t_3 = 0;

    /* "crowdposetools/_mask.pyx":185
 *             # check if list is in box format and convert it to np.ndarray
 *             isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
 *             isrle = np.all(np.array([type(obj) == dict for obj in objs]))             # <<<<<<<<<<<<<<
 *             if isbox:
 *                 objs = np.array(objs, dtype=np.double)
 */
    __pyx_t_5 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 185, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_t_12 = __Pyx_PyObject_GetAttrStr(__pyx_t_5, __pyx_n_s_all); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 185, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_12);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_4 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 185, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_7 = __Pyx_PyObject_GetAttrStr(__pyx_t_4, __pyx_n_s_array); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 185, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __pyx_t_4 = PyList_New(0); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 185, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    if (likely(PyList_CheckExact(__pyx_v_objs)) || PyTuple_CheckExact(__pyx_v_objs)) {
      __pyx_t_6 = __pyx_v_objs; __Pyx_INCREF(__pyx_t_6); __pyx_t_1 = 0;
      __pyx_t_11 = NULL;
    } else {
      __pyx_t_1 = -1; __pyx_t_6 = PyObject_GetIter(__pyx_v_objs); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 185, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __pyx_t_11 = Py_TYPE(__pyx_t_6)->tp_iternext; if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 185, __pyx_L1_error)
    }
    for (;;) {
      if (likely(!__pyx_t_11)) {
        if (likely(PyList_CheckExact(__pyx_t_6))) {
          if (__pyx_t_1 >= PyList_GET_SIZE(__pyx_t_6)) break;
          #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
          __pyx_t_10 = PyList_GET_ITEM(__pyx_t_6, __pyx_t_1); __Pyx_INCREF(__pyx_t_10); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 185, __pyx_L1_error)
          #else
          __pyx_t_10 = PySequence_ITEM(__pyx_t_6, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 185, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_10);
          #endif
        } else {
          if (__pyx_t_1 >= PyTuple_GET_SIZE(__pyx_t_6)) break;
          #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
          __pyx_t_10 = PyTuple_GET_ITEM(__pyx_t_6, __pyx_t_1); __Pyx_INCREF(__pyx_t_10); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 185, __pyx_L1_error)
          #else
          __pyx_t_10 = PySequence_ITEM(__pyx_t_6, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 185, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_10);
          #endif
        }
      } else {
        __pyx_t_10 = __pyx_t_11(__pyx_t_6);
        if (unlikely(!__pyx_t_10)) {
          PyObject* exc_type = PyErr_Occurred();
          if (exc_type) {
            if (likely(__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();
            else __PYX_ERR(0, 185, __pyx_L1_error)
          }
          break;
        }
        __Pyx_GOTREF(__pyx_t_10);
      }
      __Pyx_XDECREF_SET(__pyx_v_obj, __pyx_t_10);
      __pyx_t_10 = 0;
      __pyx_t_10 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_obj)), ((PyObject *)(&PyDict_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_10); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 185, __pyx_L1_error)
      if (unlikely(__Pyx_ListComp_Append(__pyx_t_4, (PyObject*)__pyx_t_10))) __PYX_ERR(0, 185, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
    }
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_6 = NULL;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_7))) {
      __pyx_t_6 = PyMethod_GET_SELF(__pyx_t_7);
      if (likely(__pyx_t_6)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_7);
        __Pyx_INCREF(__pyx_t_6);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_7, function);
      }
    }
    if (!__pyx_t_6) {
      __pyx_t_5 = __Pyx_PyObject_CallOneArg(__pyx_t_7, __pyx_t_4); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 185, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_5);
    } else {
      #if CYTHON_FAST_PYCALL
      if (PyFunction_Check(__pyx_t_7)) {
        PyObject *__pyx_temp[2] = {__pyx_t_6, __pyx_t_4};
        __pyx_t_5 = __Pyx_PyFunction_FastCall(__pyx_t_7, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
        __Pyx_GOTREF(__pyx_t_5);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      } else
      #endif
      #if CYTHON_FAST_PYCCALL
      if (__Pyx_PyFastCFunction_Check(__pyx_t_7)) {
        PyObject *__pyx_temp[2] = {__pyx_t_6, __pyx_t_4};
        __pyx_t_5 = __Pyx_PyCFunction_FastCall(__pyx_t_7, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
        __Pyx_GOTREF(__pyx_t_5);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      } else
      #endif
      {
        __pyx_t_10 = PyTuple_New(1+1); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_10);
        __Pyx_GIVEREF(__pyx_t_6); PyTuple_SET_ITEM(__pyx_t_10, 0, __pyx_t_6); __pyx_t_6 = NULL;
        __Pyx_GIVEREF(__pyx_t_4);
        PyTuple_SET_ITEM(__pyx_t_10, 0+1, __pyx_t_4);
        __pyx_t_4 = 0;
        __pyx_t_5 = __Pyx_PyObject_Call(__pyx_t_7, __pyx_t_10, NULL); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
      }
    }
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    __pyx_t_7 = NULL;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_12))) {
      __pyx_t_7 = PyMethod_GET_SELF(__pyx_t_12);
      if (likely(__pyx_t_7)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_12);
        __Pyx_INCREF(__pyx_t_7);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_12, function);
      }
    }
    if (!__pyx_t_7) {
      __pyx_t_3 = __Pyx_PyObject_CallOneArg(__pyx_t_12, __pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 185, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __Pyx_GOTREF(__pyx_t_3);
    } else {
      #if CYTHON_FAST_PYCALL
      if (PyFunction_Check(__pyx_t_12)) {
        PyObject *__pyx_temp[2] = {__pyx_t_7, __pyx_t_5};
        __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_12, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_7); __pyx_t_7 = 0;
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      } else
      #endif
      #if CYTHON_FAST_PYCCALL
      if (__Pyx_PyFastCFunction_Check(__pyx_t_12)) {
        PyObject *__pyx_temp[2] = {__pyx_t_7, __pyx_t_5};
        __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_12, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_7); __pyx_t_7 = 0;
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      } else
      #endif
      {
        __pyx_t_10 = PyTuple_New(1+1); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_10);
        __Pyx_GIVEREF(__pyx_t_7); PyTuple_SET_ITEM(__pyx_t_10, 0, __pyx_t_7); __pyx_t_7 = NULL;
        __Pyx_GIVEREF(__pyx_t_5);
        PyTuple_SET_ITEM(__pyx_t_10, 0+1, __pyx_t_5);
        __pyx_t_5 = 0;
        __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_12, __pyx_t_10, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 185, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
      }
    }
    __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;
    __pyx_v_isrle = __pyx_t_3;
    __pyx_t_3 = 0;

    /* "crowdposetools/_mask.pyx":186
 *             isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
 *             isrle = np.all(np.array([type(obj) == dict for obj in objs]))
 *             if isbox:             # <<<<<<<<<<<<<<
 *                 objs = np.array(objs, dtype=np.double)
 *                 if len(objs.shape) == 1:
 */
    __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_v_isbox); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 186, __pyx_L1_error)
    if (__pyx_t_2) {

      /* "crowdposetools/_mask.pyx":187
 *             isrle = np.all(np.array([type(obj) == dict for obj in objs]))
 *             if isbox:
 *                 objs = np.array(objs, dtype=np.double)             # <<<<<<<<<<<<<<
 *                 if len(objs.shape) == 1:
 *                     objs = objs.reshape((1,objs.shape[0]))
 */
      __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_12 = __Pyx_PyObject_GetAttrStr(__pyx_t_3, __pyx_n_s_array); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_12);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_INCREF(__pyx_v_objs);
      __Pyx_GIVEREF(__pyx_v_objs);
      PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_v_objs);
      __pyx_t_10 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_10);
      __pyx_t_5 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_7 = __Pyx_PyObject_GetAttrStr(__pyx_t_5, __pyx_n_s_double); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_7);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (PyDict_SetItem(__pyx_t_10, __pyx_n_s_dtype, __pyx_t_7) < 0) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
      __pyx_t_7 = __Pyx_PyObject_Call(__pyx_t_12, __pyx_t_3, __pyx_t_10); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 187, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_7);
      __Pyx_DECREF(__pyx_t_12); __pyx_t_12 = 0;
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
      __Pyx_DECREF_SET(__pyx_v_objs, __pyx_t_7);
      __pyx_t_7 = 0;

      /* "crowdposetools/_mask.pyx":188
 *             if isbox:
 *                 objs = np.array(objs, dtype=np.double)
 *                 if len(objs.shape) == 1:             # <<<<<<<<<<<<<<
 *                     objs = objs.reshape((1,objs.shape[0]))
 *             elif isrle:
 */
      __pyx_t_7 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_shape); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 188, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_7);
      __pyx_t_1 = PyObject_Length(__pyx_t_7); if (unlikely(__pyx_t_1 == ((Py_ssize_t)-1))) __PYX_ERR(0, 188, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
      __pyx_t_2 = ((__pyx_t_1 == 1) != 0);
      if (__pyx_t_2) {

        /* "crowdposetools/_mask.pyx":189
 *                 objs = np.array(objs, dtype=np.double)
 *                 if len(objs.shape) == 1:
 *                     objs = objs.reshape((1,objs.shape[0]))             # <<<<<<<<<<<<<<
 *             elif isrle:
 *                 objs = _frString(objs)
 */
        __pyx_t_10 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_reshape); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 189, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_10);
        __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_v_objs, __pyx_n_s_shape); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 189, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __pyx_t_12 = __Pyx_GetItemInt(__pyx_t_3, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_12)) __PYX_ERR(0, 189, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_12);
        __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
        __pyx_t_3 = PyTuple_New(2); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 189, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_INCREF(__pyx_int_1);
        __Pyx_GIVEREF(__pyx_int_1);
        PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_int_1);
        __Pyx_GIVEREF(__pyx_t_12);
        PyTuple_SET_ITEM(__pyx_t_3, 1, __pyx_t_12);
        __pyx_t_12 = 0;
        __pyx_t_12 = NULL;
        if (CYTHON_UNPACK_METHODS && likely(PyMethod_Check(__pyx_t_10))) {
          __pyx_t_12 = PyMethod_GET_SELF(__pyx_t_10);
          if (likely(__pyx_t_12)) {
            PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_10);
            __Pyx_INCREF(__pyx_t_12);
            __Pyx_INCREF(function);
            __Pyx_DECREF_SET(__pyx_t_10, function);
          }
        }
        if (!__pyx_t_12) {
          __pyx_t_7 = __Pyx_PyObject_CallOneArg(__pyx_t_10, __pyx_t_3); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 189, __pyx_L1_error)
          __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
          __Pyx_GOTREF(__pyx_t_7);
        } else {
          #if CYTHON_FAST_PYCALL
          if (PyFunction_Check(__pyx_t_10)) {
            PyObject *__pyx_temp[2] = {__pyx_t_12, __pyx_t_3};
            __pyx_t_7 = __Pyx_PyFunction_FastCall(__pyx_t_10, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 189, __pyx_L1_error)
            __Pyx_XDECREF(__pyx_t_12); __pyx_t_12 = 0;
            __Pyx_GOTREF(__pyx_t_7);
            __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
          } else
          #endif
          #if CYTHON_FAST_PYCCALL
          if (__Pyx_PyFastCFunction_Check(__pyx_t_10)) {
            PyObject *__pyx_temp[2] = {__pyx_t_12, __pyx_t_3};
            __pyx_t_7 = __Pyx_PyCFunction_FastCall(__pyx_t_10, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 189, __pyx_L1_error)
            __Pyx_XDECREF(__pyx_t_12); __pyx_t_12 = 0;
            __Pyx_GOTREF(__pyx_t_7);
            __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
          } else
          #endif
          {
            __pyx_t_5 = PyTuple_New(1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 189, __pyx_L1_error)
            __Pyx_GOTREF(__pyx_t_5);
            __Pyx_GIVEREF(__pyx_t_12); PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_12); __pyx_t_12 = NULL;
            __Pyx_GIVEREF(__pyx_t_3);
            PyTuple_SET_ITEM(__pyx_t_5, 0+1, __pyx_t_3);
            __pyx_t_3 = 0;
            __pyx_t_7 = __Pyx_PyObject_Call(__pyx_t_10, __pyx_t_5, NULL); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 189, __pyx_L1_error)
            __Pyx_GOTREF(__pyx_t_7);
            __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
          }
        }
        __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
        __Pyx_DECREF_SET(__pyx_v_objs, __pyx_t_7);
        __pyx_t_7 = 0;

        /* "crowdposetools/_mask.pyx":188
 *             if isbox:
 *                 objs = np.array(objs, dtype=np.double)
 *                 if len(objs.shape) == 1:             # <<<<<<<<<<<<<<
 *                     objs = objs.reshape((1,objs.shape[0]))
 *             elif isrle:
 */
      }

      /* "crowdposetools/_mask.pyx":186
 *             isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
 *             isrle = np.all(np.array([type(obj) == dict for obj in objs]))
 *             if isbox:             # <<<<<<<<<<<<<<
 *                 objs = np.array(objs, dtype=np.double)
 *                 if len(objs.shape) == 1:
 */
      goto __pyx_L16;
    }

    /* "crowdposetools/_mask.pyx":190
 *                 if len(objs.shape) == 1:
 *                     objs = objs.reshape((1,objs.shape[0]))
 *             elif isrle:             # <<<<<<<<<<<<<<
 *                 objs = _frString(objs)
 *             else:
 */
    __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_v_isrle); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 190, __pyx_L1_error)
    if (likely(__pyx_t_2)) {

      /* "crowdposetools/_mask.pyx":191
 *                     objs = objs.reshape((1,objs.shape[0]))
 *             elif isrle:
 *                 objs = _frString(objs)             # <<<<<<<<<<<<<<
 *             else:
 *                 raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
 */
      __pyx_t_10 = __Pyx_GetModuleGlobalName(__pyx_n_s_frString); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 191, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_10);
      __pyx_t_5 = NULL;
      if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_10))) {
        __pyx_t_5 = PyMethod_GET_SELF(__pyx_t_10);
        if (likely(__pyx_t_5)) {
          PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_10);
          __Pyx_INCREF(__pyx_t_5);
          __Pyx_INCREF(function);
          __Pyx_DECREF_SET(__pyx_t_10, function);
        }
      }
      if (!__pyx_t_5) {
        __pyx_t_7 = __Pyx_PyObject_CallOneArg(__pyx_t_10, __pyx_v_objs); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 191, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_7);
      } else {
        #if CYTHON_FAST_PYCALL
        if (PyFunction_Check(__pyx_t_10)) {
          PyObject *__pyx_temp[2] = {__pyx_t_5, __pyx_v_objs};
          __pyx_t_7 = __Pyx_PyFunction_FastCall(__pyx_t_10, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 191, __pyx_L1_error)
          __Pyx_XDECREF(__pyx_t_5); __pyx_t_5 = 0;
          __Pyx_GOTREF(__pyx_t_7);
        } else
        #endif
        #if CYTHON_FAST_PYCCALL
        if (__Pyx_PyFastCFunction_Check(__pyx_t_10)) {
          PyObject *__pyx_temp[2] = {__pyx_t_5, __pyx_v_objs};
          __pyx_t_7 = __Pyx_PyCFunction_FastCall(__pyx_t_10, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 191, __pyx_L1_error)
          __Pyx_XDECREF(__pyx_t_5); __pyx_t_5 = 0;
          __Pyx_GOTREF(__pyx_t_7);
        } else
        #endif
        {
          __pyx_t_3 = PyTuple_New(1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 191, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_3);
          __Pyx_GIVEREF(__pyx_t_5); PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_5); __pyx_t_5 = NULL;
          __Pyx_INCREF(__pyx_v_objs);
          __Pyx_GIVEREF(__pyx_v_objs);
          PyTuple_SET_ITEM(__pyx_t_3, 0+1, __pyx_v_objs);
          __pyx_t_7 = __Pyx_PyObject_Call(__pyx_t_10, __pyx_t_3, NULL); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 191, __pyx_L1_error)
          __Pyx_GOTREF(__pyx_t_7);
          __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
        }
      }
      __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
      __Pyx_DECREF_SET(__pyx_v_objs, __pyx_t_7);
      __pyx_t_7 = 0;

      /* "crowdposetools/_mask.pyx":190
 *                 if len(objs.shape) == 1:
 *                     objs = objs.reshape((1,objs.shape[0]))
 *             elif isrle:             # <<<<<<<<<<<<<<
 *                 objs = _frString(objs)
 *             else:
 */
      goto __pyx_L16;
    }

    /* "crowdposetools/_mask.pyx":193
 *                 objs = _frString(objs)
 *             else:
 *                 raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')             # <<<<<<<<<<<<<<
 *         else:
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
 */
    /*else*/ {
      __pyx_t_7 = __Pyx_PyObject_Call(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])), __pyx_tuple__9, NULL); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 193, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_7);
      __Pyx_Raise(__pyx_t_7, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
      __PYX_ERR(0, 193, __pyx_L1_error)
    }
    __pyx_L16:;

    /* "crowdposetools/_mask.pyx":182
 *                 raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
 *             objs = objs.astype(np.double)
 *         elif type(objs) == list:             # <<<<<<<<<<<<<<
 *             # check if list is in box format and convert it to np.ndarray
 *             isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
 */
    goto __pyx_L4;
  }

  /* "crowdposetools/_mask.pyx":195
 *                 raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
 *         else:
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')             # <<<<<<<<<<<<<<
 *         return objs
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 */
  /*else*/ {
    __pyx_t_7 = __Pyx_PyObject_Call(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])), __pyx_tuple__10, NULL); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 195, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __Pyx_Raise(__pyx_t_7, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    __PYX_ERR(0, 195, __pyx_L1_error)
  }
  __pyx_L4:;

  /* "crowdposetools/_mask.pyx":196
 *         else:
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
 *         return objs             # <<<<<<<<<<<<<<
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_objs);
  __pyx_r = __pyx_v_objs;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":172
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):
 *     def _preproc(objs):             # <<<<<<<<<<<<<<
 *         if len(objs) == 0:
 *             return objs
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_10);
  __Pyx_XDECREF(__pyx_t_12);
  __Pyx_XDECREF(__pyx_t_14);
  __Pyx_AddTraceback("crowdposetools._mask.iou._preproc", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF(__pyx_v_isbox);
  __Pyx_XDECREF(__pyx_v_isrle);
  __Pyx_XDECREF(__pyx_v_obj);
  __Pyx_XDECREF(__pyx_v_objs);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":197
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
 *         return objs
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_3_rleIou(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_3iou_3_rleIou = {"_rleIou", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_3iou_3_rleIou, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_3_rleIou(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_dt = 0;
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_gt = 0;
  PyArrayObject *__pyx_v_iscrowd = 0;
  siz __pyx_v_m;
  siz __pyx_v_n;
  PyArrayObject *__pyx_v__iou = 0;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_rleIou (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_dt,&__pyx_n_s_gt,&__pyx_n_s_iscrowd,&__pyx_n_s_m,&__pyx_n_s_n,&__pyx_n_s_iou,0};
    PyObject* values[6] = {0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        CYTHON_FALLTHROUGH;
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        CYTHON_FALLTHROUGH;
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        CYTHON_FALLTHROUGH;
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_dt)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_gt)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_rleIou", 1, 6, 6, 1); __PYX_ERR(0, 197, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_iscrowd)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_rleIou", 1, 6, 6, 2); __PYX_ERR(0, 197, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  3:
        if (likely((values[3] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_m)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_rleIou", 1, 6, 6, 3); __PYX_ERR(0, 197, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  4:
        if (likely((values[4] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_n)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_rleIou", 1, 6, 6, 4); __PYX_ERR(0, 197, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  5:
        if (likely((values[5] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_iou)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_rleIou", 1, 6, 6, 5); __PYX_ERR(0, 197, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "_rleIou") < 0)) __PYX_ERR(0, 197, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 6) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
      values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
      values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
    }
    __pyx_v_dt = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)values[0]);
    __pyx_v_gt = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)values[1]);
    __pyx_v_iscrowd = ((PyArrayObject *)values[2]);
    __pyx_v_m = __Pyx_PyInt_As_siz(values[3]); if (unlikely((__pyx_v_m == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 197, __pyx_L3_error)
    __pyx_v_n = __Pyx_PyInt_As_siz(values[4]); if (unlikely((__pyx_v_n == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 197, __pyx_L3_error)
    __pyx_v__iou = ((PyArrayObject *)values[5]);
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("_rleIou", 1, 6, 6, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 197, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.iou._rleIou", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_dt), __pyx_ptype_14crowdposetools_5_mask_RLEs, 1, "dt", 0))) __PYX_ERR(0, 197, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_gt), __pyx_ptype_14crowdposetools_5_mask_RLEs, 1, "gt", 0))) __PYX_ERR(0, 197, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_iscrowd), __pyx_ptype_5numpy_ndarray, 1, "iscrowd", 0))) __PYX_ERR(0, 197, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v__iou), __pyx_ptype_5numpy_ndarray, 1, "_iou", 0))) __PYX_ERR(0, 197, __pyx_L1_error)
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_3iou_2_rleIou(__pyx_self, __pyx_v_dt, __pyx_v_gt, __pyx_v_iscrowd, __pyx_v_m, __pyx_v_n, __pyx_v__iou);

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou_2_rleIou(CYTHON_UNUSED PyObject *__pyx_self, struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_dt, struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_gt, PyArrayObject *__pyx_v_iscrowd, siz __pyx_v_m, siz __pyx_v_n, PyArrayObject *__pyx_v__iou) {
  __Pyx_LocalBuf_ND __pyx_pybuffernd__iou;
  __Pyx_Buffer __pyx_pybuffer__iou;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_iscrowd;
  __Pyx_Buffer __pyx_pybuffer_iscrowd;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_rleIou", 0);
  __pyx_pybuffer_iscrowd.pybuffer.buf = NULL;
  __pyx_pybuffer_iscrowd.refcount = 0;
  __pyx_pybuffernd_iscrowd.data = NULL;
  __pyx_pybuffernd_iscrowd.rcbuffer = &__pyx_pybuffer_iscrowd;
  __pyx_pybuffer__iou.pybuffer.buf = NULL;
  __pyx_pybuffer__iou.refcount = 0;
  __pyx_pybuffernd__iou.data = NULL;
  __pyx_pybuffernd__iou.rcbuffer = &__pyx_pybuffer__iou;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer, (PyObject*)__pyx_v_iscrowd, &__Pyx_TypeInfo_nn___pyx_t_5numpy_uint8_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) __PYX_ERR(0, 197, __pyx_L1_error)
  }
  __pyx_pybuffernd_iscrowd.diminfo[0].strides = __pyx_pybuffernd_iscrowd.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_iscrowd.diminfo[0].shape = __pyx_pybuffernd_iscrowd.rcbuffer->pybuffer.shape[0];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd__iou.rcbuffer->pybuffer, (PyObject*)__pyx_v__iou, &__Pyx_TypeInfo_nn___pyx_t_5numpy_double_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) __PYX_ERR(0, 197, __pyx_L1_error)
  }
  __pyx_pybuffernd__iou.diminfo[0].strides = __pyx_pybuffernd__iou.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd__iou.diminfo[0].shape = __pyx_pybuffernd__iou.rcbuffer->pybuffer.shape[0];

  /* "crowdposetools/_mask.pyx":198
 *         return objs
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )             # <<<<<<<<<<<<<<
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 */
  rleIou(((RLE *)__pyx_v_dt->_R), ((RLE *)__pyx_v_gt->_R), __pyx_v_m, __pyx_v_n, ((byte *)__pyx_v_iscrowd->data), ((double *)__pyx_v__iou->data));

  /* "crowdposetools/_mask.pyx":197
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
 *         return objs
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 */

  /* function exit code */
  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd__iou.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("crowdposetools._mask.iou._rleIou", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd__iou.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":199
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_5_bbIou(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_3iou_5_bbIou = {"_bbIou", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_3iou_5_bbIou, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_5_bbIou(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyArrayObject *__pyx_v_dt = 0;
  PyArrayObject *__pyx_v_gt = 0;
  PyArrayObject *__pyx_v_iscrowd = 0;
  siz __pyx_v_m;
  siz __pyx_v_n;
  PyArrayObject *__pyx_v__iou = 0;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_bbIou (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_dt,&__pyx_n_s_gt,&__pyx_n_s_iscrowd,&__pyx_n_s_m,&__pyx_n_s_n,&__pyx_n_s_iou,0};
    PyObject* values[6] = {0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        CYTHON_FALLTHROUGH;
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        CYTHON_FALLTHROUGH;
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        CYTHON_FALLTHROUGH;
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_dt)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_gt)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_bbIou", 1, 6, 6, 1); __PYX_ERR(0, 199, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_iscrowd)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_bbIou", 1, 6, 6, 2); __PYX_ERR(0, 199, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  3:
        if (likely((values[3] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_m)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_bbIou", 1, 6, 6, 3); __PYX_ERR(0, 199, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  4:
        if (likely((values[4] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_n)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_bbIou", 1, 6, 6, 4); __PYX_ERR(0, 199, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  5:
        if (likely((values[5] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_iou)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("_bbIou", 1, 6, 6, 5); __PYX_ERR(0, 199, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "_bbIou") < 0)) __PYX_ERR(0, 199, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 6) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
      values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
      values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
    }
    __pyx_v_dt = ((PyArrayObject *)values[0]);
    __pyx_v_gt = ((PyArrayObject *)values[1]);
    __pyx_v_iscrowd = ((PyArrayObject *)values[2]);
    __pyx_v_m = __Pyx_PyInt_As_siz(values[3]); if (unlikely((__pyx_v_m == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 199, __pyx_L3_error)
    __pyx_v_n = __Pyx_PyInt_As_siz(values[4]); if (unlikely((__pyx_v_n == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 199, __pyx_L3_error)
    __pyx_v__iou = ((PyArrayObject *)values[5]);
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("_bbIou", 1, 6, 6, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 199, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.iou._bbIou", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_dt), __pyx_ptype_5numpy_ndarray, 1, "dt", 0))) __PYX_ERR(0, 199, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_gt), __pyx_ptype_5numpy_ndarray, 1, "gt", 0))) __PYX_ERR(0, 199, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_iscrowd), __pyx_ptype_5numpy_ndarray, 1, "iscrowd", 0))) __PYX_ERR(0, 199, __pyx_L1_error)
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v__iou), __pyx_ptype_5numpy_ndarray, 1, "_iou", 0))) __PYX_ERR(0, 199, __pyx_L1_error)
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_3iou_4_bbIou(__pyx_self, __pyx_v_dt, __pyx_v_gt, __pyx_v_iscrowd, __pyx_v_m, __pyx_v_n, __pyx_v__iou);

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou_4_bbIou(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_dt, PyArrayObject *__pyx_v_gt, PyArrayObject *__pyx_v_iscrowd, siz __pyx_v_m, siz __pyx_v_n, PyArrayObject *__pyx_v__iou) {
  __Pyx_LocalBuf_ND __pyx_pybuffernd__iou;
  __Pyx_Buffer __pyx_pybuffer__iou;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_dt;
  __Pyx_Buffer __pyx_pybuffer_dt;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_gt;
  __Pyx_Buffer __pyx_pybuffer_gt;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_iscrowd;
  __Pyx_Buffer __pyx_pybuffer_iscrowd;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_bbIou", 0);
  __pyx_pybuffer_dt.pybuffer.buf = NULL;
  __pyx_pybuffer_dt.refcount = 0;
  __pyx_pybuffernd_dt.data = NULL;
  __pyx_pybuffernd_dt.rcbuffer = &__pyx_pybuffer_dt;
  __pyx_pybuffer_gt.pybuffer.buf = NULL;
  __pyx_pybuffer_gt.refcount = 0;
  __pyx_pybuffernd_gt.data = NULL;
  __pyx_pybuffernd_gt.rcbuffer = &__pyx_pybuffer_gt;
  __pyx_pybuffer_iscrowd.pybuffer.buf = NULL;
  __pyx_pybuffer_iscrowd.refcount = 0;
  __pyx_pybuffernd_iscrowd.data = NULL;
  __pyx_pybuffernd_iscrowd.rcbuffer = &__pyx_pybuffer_iscrowd;
  __pyx_pybuffer__iou.pybuffer.buf = NULL;
  __pyx_pybuffer__iou.refcount = 0;
  __pyx_pybuffernd__iou.data = NULL;
  __pyx_pybuffernd__iou.rcbuffer = &__pyx_pybuffer__iou;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_dt.rcbuffer->pybuffer, (PyObject*)__pyx_v_dt, &__Pyx_TypeInfo_nn___pyx_t_5numpy_double_t, PyBUF_FORMAT| PyBUF_STRIDES, 2, 0, __pyx_stack) == -1)) __PYX_ERR(0, 199, __pyx_L1_error)
  }
  __pyx_pybuffernd_dt.diminfo[0].strides = __pyx_pybuffernd_dt.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_dt.diminfo[0].shape = __pyx_pybuffernd_dt.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_dt.diminfo[1].strides = __pyx_pybuffernd_dt.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_dt.diminfo[1].shape = __pyx_pybuffernd_dt.rcbuffer->pybuffer.shape[1];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_gt.rcbuffer->pybuffer, (PyObject*)__pyx_v_gt, &__Pyx_TypeInfo_nn___pyx_t_5numpy_double_t, PyBUF_FORMAT| PyBUF_STRIDES, 2, 0, __pyx_stack) == -1)) __PYX_ERR(0, 199, __pyx_L1_error)
  }
  __pyx_pybuffernd_gt.diminfo[0].strides = __pyx_pybuffernd_gt.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_gt.diminfo[0].shape = __pyx_pybuffernd_gt.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_gt.diminfo[1].strides = __pyx_pybuffernd_gt.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_gt.diminfo[1].shape = __pyx_pybuffernd_gt.rcbuffer->pybuffer.shape[1];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer, (PyObject*)__pyx_v_iscrowd, &__Pyx_TypeInfo_nn___pyx_t_5numpy_uint8_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) __PYX_ERR(0, 199, __pyx_L1_error)
  }
  __pyx_pybuffernd_iscrowd.diminfo[0].strides = __pyx_pybuffernd_iscrowd.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_iscrowd.diminfo[0].shape = __pyx_pybuffernd_iscrowd.rcbuffer->pybuffer.shape[0];
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd__iou.rcbuffer->pybuffer, (PyObject*)__pyx_v__iou, &__Pyx_TypeInfo_nn___pyx_t_5numpy_double_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) __PYX_ERR(0, 199, __pyx_L1_error)
  }
  __pyx_pybuffernd__iou.diminfo[0].strides = __pyx_pybuffernd__iou.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd__iou.diminfo[0].shape = __pyx_pybuffernd__iou.rcbuffer->pybuffer.shape[0];

  /* "crowdposetools/_mask.pyx":200
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )             # <<<<<<<<<<<<<<
 *     def _len(obj):
 *         cdef siz N = 0
 */
  bbIou(((BB)__pyx_v_dt->data), ((BB)__pyx_v_gt->data), __pyx_v_m, __pyx_v_n, ((byte *)__pyx_v_iscrowd->data), ((double *)__pyx_v__iou->data));

  /* "crowdposetools/_mask.pyx":199
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):
 */

  /* function exit code */
  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd__iou.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_dt.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_gt.rcbuffer->pybuffer);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("crowdposetools._mask.iou._bbIou", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd__iou.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_dt.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_gt.rcbuffer->pybuffer);
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":201
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):             # <<<<<<<<<<<<<<
 *         cdef siz N = 0
 *         if type(obj) == RLEs:
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_7_len(PyObject *__pyx_self, PyObject *__pyx_v_obj); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_3iou_7_len = {"_len", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_3iou_7_len, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_3iou_7_len(PyObject *__pyx_self, PyObject *__pyx_v_obj) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("_len (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_3iou_6_len(__pyx_self, ((PyObject *)__pyx_v_obj));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_3iou_6_len(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_obj) {
  siz __pyx_v_N;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_t_2;
  siz __pyx_t_3;
  Py_ssize_t __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  __Pyx_RefNannySetupContext("_len", 0);

  /* "crowdposetools/_mask.pyx":202
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):
 *         cdef siz N = 0             # <<<<<<<<<<<<<<
 *         if type(obj) == RLEs:
 *             N = obj.n
 */
  __pyx_v_N = 0;

  /* "crowdposetools/_mask.pyx":203
 *     def _len(obj):
 *         cdef siz N = 0
 *         if type(obj) == RLEs:             # <<<<<<<<<<<<<<
 *             N = obj.n
 *         elif len(obj)==0:
 */
  __pyx_t_1 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_obj)), ((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), Py_EQ); __Pyx_XGOTREF(__pyx_t_1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 203, __pyx_L1_error)
  __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_t_1); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 203, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":204
 *         cdef siz N = 0
 *         if type(obj) == RLEs:
 *             N = obj.n             # <<<<<<<<<<<<<<
 *         elif len(obj)==0:
 *             pass
 */
    __pyx_t_1 = __Pyx_PyObject_GetAttrStr(__pyx_v_obj, __pyx_n_s_n); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 204, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_3 = __Pyx_PyInt_As_siz(__pyx_t_1); if (unlikely((__pyx_t_3 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 204, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_N = __pyx_t_3;

    /* "crowdposetools/_mask.pyx":203
 *     def _len(obj):
 *         cdef siz N = 0
 *         if type(obj) == RLEs:             # <<<<<<<<<<<<<<
 *             N = obj.n
 *         elif len(obj)==0:
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":205
 *         if type(obj) == RLEs:
 *             N = obj.n
 *         elif len(obj)==0:             # <<<<<<<<<<<<<<
 *             pass
 *         elif type(obj) == np.ndarray:
 */
  __pyx_t_4 = PyObject_Length(__pyx_v_obj); if (unlikely(__pyx_t_4 == ((Py_ssize_t)-1))) __PYX_ERR(0, 205, __pyx_L1_error)
  __pyx_t_2 = ((__pyx_t_4 == 0) != 0);
  if (__pyx_t_2) {
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":207
 *         elif len(obj)==0:
 *             pass
 *         elif type(obj) == np.ndarray:             # <<<<<<<<<<<<<<
 *             N = obj.shape[0]
 *         return N
 */
  __pyx_t_1 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_obj)), ((PyObject *)__pyx_ptype_5numpy_ndarray), Py_EQ); __Pyx_XGOTREF(__pyx_t_1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 207, __pyx_L1_error)
  __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_t_1); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 207, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":208
 *             pass
 *         elif type(obj) == np.ndarray:
 *             N = obj.shape[0]             # <<<<<<<<<<<<<<
 *         return N
 *     # convert iscrowd to numpy array
 */
    __pyx_t_1 = __Pyx_PyObject_GetAttrStr(__pyx_v_obj, __pyx_n_s_shape); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 208, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_5 = __Pyx_GetItemInt(__pyx_t_1, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 208, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_3 = __Pyx_PyInt_As_siz(__pyx_t_5); if (unlikely((__pyx_t_3 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 208, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_v_N = __pyx_t_3;

    /* "crowdposetools/_mask.pyx":207
 *         elif len(obj)==0:
 *             pass
 *         elif type(obj) == np.ndarray:             # <<<<<<<<<<<<<<
 *             N = obj.shape[0]
 *         return N
 */
  }
  __pyx_L3:;

  /* "crowdposetools/_mask.pyx":209
 *         elif type(obj) == np.ndarray:
 *             N = obj.shape[0]
 *         return N             # <<<<<<<<<<<<<<
 *     # convert iscrowd to numpy array
 *     cdef np.ndarray[np.uint8_t, ndim=1] iscrowd = np.array(pyiscrowd, dtype=np.uint8)
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_5 = __Pyx_PyInt_From_siz(__pyx_v_N); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 209, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_r = __pyx_t_5;
  __pyx_t_5 = 0;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":201
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):             # <<<<<<<<<<<<<<
 *         cdef siz N = 0
 *         if type(obj) == RLEs:
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_AddTraceback("crowdposetools._mask.iou._len", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":171
 * 
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):             # <<<<<<<<<<<<<<
 *     def _preproc(objs):
 *         if len(objs) == 0:
 */

static PyObject *__pyx_pf_14crowdposetools_5_mask_12iou(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_dt, PyObject *__pyx_v_gt, PyObject *__pyx_v_pyiscrowd) {
  PyObject *__pyx_v__preproc = 0;
  PyObject *__pyx_v__rleIou = 0;
  PyObject *__pyx_v__bbIou = 0;
  PyObject *__pyx_v__len = 0;
  PyArrayObject *__pyx_v_iscrowd = 0;
  siz __pyx_v_m;
  siz __pyx_v_n;
  double *__pyx_v__iou;
  npy_intp __pyx_v_shape[1];
  PyObject *__pyx_v__iouFun = NULL;
  PyObject *__pyx_v_iou = NULL;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_iscrowd;
  __Pyx_Buffer __pyx_pybuffer_iscrowd;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  PyArrayObject *__pyx_t_6 = NULL;
  siz __pyx_t_7;
  int __pyx_t_8;
  int __pyx_t_9;
  int __pyx_t_10;
  PyObject *__pyx_t_11 = NULL;
  __Pyx_RefNannySetupContext("iou", 0);
  __Pyx_INCREF(__pyx_v_dt);
  __Pyx_INCREF(__pyx_v_gt);
  __pyx_pybuffer_iscrowd.pybuffer.buf = NULL;
  __pyx_pybuffer_iscrowd.refcount = 0;
  __pyx_pybuffernd_iscrowd.data = NULL;
  __pyx_pybuffernd_iscrowd.rcbuffer = &__pyx_pybuffer_iscrowd;

  /* "crowdposetools/_mask.pyx":172
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):
 *     def _preproc(objs):             # <<<<<<<<<<<<<<
 *         if len(objs) == 0:
 *             return objs
 */
  __pyx_t_1 = __Pyx_CyFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_3iou_1_preproc, 0, __pyx_n_s_iou_locals__preproc, NULL, __pyx_n_s_crowdposetools__mask, __pyx_d, ((PyObject *)__pyx_codeobj__12)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 172, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_v__preproc = __pyx_t_1;
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":197
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
 *         return objs
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 */
  __pyx_t_1 = __Pyx_CyFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_3iou_3_rleIou, 0, __pyx_n_s_iou_locals__rleIou, NULL, __pyx_n_s_crowdposetools__mask, __pyx_d, ((PyObject *)__pyx_codeobj__14)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 197, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_v__rleIou = __pyx_t_1;
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":199
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):
 */
  __pyx_t_1 = __Pyx_CyFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_3iou_5_bbIou, 0, __pyx_n_s_iou_locals__bbIou, NULL, __pyx_n_s_crowdposetools__mask, __pyx_d, ((PyObject *)__pyx_codeobj__16)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 199, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_v__bbIou = __pyx_t_1;
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":201
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):             # <<<<<<<<<<<<<<
 *         cdef siz N = 0
 *         if type(obj) == RLEs:
 */
  __pyx_t_1 = __Pyx_CyFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_3iou_7_len, 0, __pyx_n_s_iou_locals__len, NULL, __pyx_n_s_crowdposetools__mask, __pyx_d, ((PyObject *)__pyx_codeobj__18)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 201, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_v__len = __pyx_t_1;
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":211
 *         return N
 *     # convert iscrowd to numpy array
 *     cdef np.ndarray[np.uint8_t, ndim=1] iscrowd = np.array(pyiscrowd, dtype=np.uint8)             # <<<<<<<<<<<<<<
 *     # simple type checking
 *     cdef siz m, n
 */
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_array); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyTuple_New(1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_INCREF(__pyx_v_pyiscrowd);
  __Pyx_GIVEREF(__pyx_v_pyiscrowd);
  PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_v_pyiscrowd);
  __pyx_t_3 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_4 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_5 = __Pyx_PyObject_GetAttrStr(__pyx_t_4, __pyx_n_s_uint8); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  if (PyDict_SetItem(__pyx_t_3, __pyx_n_s_dtype, __pyx_t_5) < 0) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_1, __pyx_t_3); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 211, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (!(likely(((__pyx_t_5) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_5, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 211, __pyx_L1_error)
  __pyx_t_6 = ((PyArrayObject *)__pyx_t_5);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer, (PyObject*)__pyx_t_6, &__Pyx_TypeInfo_nn___pyx_t_5numpy_uint8_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_iscrowd = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_pybuffernd_iscrowd.rcbuffer->pybuffer.buf = NULL;
      __PYX_ERR(0, 211, __pyx_L1_error)
    } else {__pyx_pybuffernd_iscrowd.diminfo[0].strides = __pyx_pybuffernd_iscrowd.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_iscrowd.diminfo[0].shape = __pyx_pybuffernd_iscrowd.rcbuffer->pybuffer.shape[0];
    }
  }
  __pyx_t_6 = 0;
  __pyx_v_iscrowd = ((PyArrayObject *)__pyx_t_5);
  __pyx_t_5 = 0;

  /* "crowdposetools/_mask.pyx":214
 *     # simple type checking
 *     cdef siz m, n
 *     dt = _preproc(dt)             # <<<<<<<<<<<<<<
 *     gt = _preproc(gt)
 *     m = _len(dt)
 */
  __pyx_t_5 = __pyx_pf_14crowdposetools_5_mask_3iou__preproc(__pyx_v__preproc, __pyx_v_dt); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 214, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF_SET(__pyx_v_dt, __pyx_t_5);
  __pyx_t_5 = 0;

  /* "crowdposetools/_mask.pyx":215
 *     cdef siz m, n
 *     dt = _preproc(dt)
 *     gt = _preproc(gt)             # <<<<<<<<<<<<<<
 *     m = _len(dt)
 *     n = _len(gt)
 */
  __pyx_t_5 = __pyx_pf_14crowdposetools_5_mask_3iou__preproc(__pyx_v__preproc, __pyx_v_gt); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 215, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF_SET(__pyx_v_gt, __pyx_t_5);
  __pyx_t_5 = 0;

  /* "crowdposetools/_mask.pyx":216
 *     dt = _preproc(dt)
 *     gt = _preproc(gt)
 *     m = _len(dt)             # <<<<<<<<<<<<<<
 *     n = _len(gt)
 *     if m == 0 or n == 0:
 */
  __pyx_t_5 = __pyx_pf_14crowdposetools_5_mask_3iou_6_len(__pyx_v__len, __pyx_v_dt); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 216, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_7 = __Pyx_PyInt_As_siz(__pyx_t_5); if (unlikely((__pyx_t_7 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 216, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_v_m = __pyx_t_7;

  /* "crowdposetools/_mask.pyx":217
 *     gt = _preproc(gt)
 *     m = _len(dt)
 *     n = _len(gt)             # <<<<<<<<<<<<<<
 *     if m == 0 or n == 0:
 *         return []
 */
  __pyx_t_5 = __pyx_pf_14crowdposetools_5_mask_3iou_6_len(__pyx_v__len, __pyx_v_gt); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 217, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_7 = __Pyx_PyInt_As_siz(__pyx_t_5); if (unlikely((__pyx_t_7 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 217, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_v_n = __pyx_t_7;

  /* "crowdposetools/_mask.pyx":218
 *     m = _len(dt)
 *     n = _len(gt)
 *     if m == 0 or n == 0:             # <<<<<<<<<<<<<<
 *         return []
 *     if not type(dt) == type(gt):
 */
  __pyx_t_9 = ((__pyx_v_m == 0) != 0);
  if (!__pyx_t_9) {
  } else {
    __pyx_t_8 = __pyx_t_9;
    goto __pyx_L4_bool_binop_done;
  }
  __pyx_t_9 = ((__pyx_v_n == 0) != 0);
  __pyx_t_8 = __pyx_t_9;
  __pyx_L4_bool_binop_done:;
  if (__pyx_t_8) {

    /* "crowdposetools/_mask.pyx":219
 *     n = _len(gt)
 *     if m == 0 or n == 0:
 *         return []             # <<<<<<<<<<<<<<
 *     if not type(dt) == type(gt):
 *         raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')
 */
    __Pyx_XDECREF(__pyx_r);
    __pyx_t_5 = PyList_New(0); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 219, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_r = __pyx_t_5;
    __pyx_t_5 = 0;
    goto __pyx_L0;

    /* "crowdposetools/_mask.pyx":218
 *     m = _len(dt)
 *     n = _len(gt)
 *     if m == 0 or n == 0:             # <<<<<<<<<<<<<<
 *         return []
 *     if not type(dt) == type(gt):
 */
  }

  /* "crowdposetools/_mask.pyx":220
 *     if m == 0 or n == 0:
 *         return []
 *     if not type(dt) == type(gt):             # <<<<<<<<<<<<<<
 *         raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')
 * 
 */
  __pyx_t_5 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_dt)), ((PyObject *)Py_TYPE(__pyx_v_gt)), Py_EQ); __Pyx_XGOTREF(__pyx_t_5); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 220, __pyx_L1_error)
  __pyx_t_8 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_8 < 0)) __PYX_ERR(0, 220, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_9 = ((!__pyx_t_8) != 0);
  if (unlikely(__pyx_t_9)) {

    /* "crowdposetools/_mask.pyx":221
 *         return []
 *     if not type(dt) == type(gt):
 *         raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')             # <<<<<<<<<<<<<<
 * 
 *     # define local variables
 */
    __pyx_t_5 = __Pyx_PyObject_Call(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])), __pyx_tuple__19, NULL); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 221, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_Raise(__pyx_t_5, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __PYX_ERR(0, 221, __pyx_L1_error)

    /* "crowdposetools/_mask.pyx":220
 *     if m == 0 or n == 0:
 *         return []
 *     if not type(dt) == type(gt):             # <<<<<<<<<<<<<<
 *         raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')
 * 
 */
  }

  /* "crowdposetools/_mask.pyx":224
 * 
 *     # define local variables
 *     cdef double* _iou = <double*> 0             # <<<<<<<<<<<<<<
 *     cdef np.npy_intp shape[1]
 *     # check type and assign iou function
 */
  __pyx_v__iou = ((double *)0);

  /* "crowdposetools/_mask.pyx":227
 *     cdef np.npy_intp shape[1]
 *     # check type and assign iou function
 *     if type(dt) == RLEs:             # <<<<<<<<<<<<<<
 *         _iouFun = _rleIou
 *     elif type(dt) == np.ndarray:
 */
  __pyx_t_5 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_dt)), ((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), Py_EQ); __Pyx_XGOTREF(__pyx_t_5); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 227, __pyx_L1_error)
  __pyx_t_9 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_9 < 0)) __PYX_ERR(0, 227, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  if (__pyx_t_9) {

    /* "crowdposetools/_mask.pyx":228
 *     # check type and assign iou function
 *     if type(dt) == RLEs:
 *         _iouFun = _rleIou             # <<<<<<<<<<<<<<
 *     elif type(dt) == np.ndarray:
 *         _iouFun = _bbIou
 */
    __Pyx_INCREF(__pyx_v__rleIou);
    __pyx_v__iouFun = __pyx_v__rleIou;

    /* "crowdposetools/_mask.pyx":227
 *     cdef np.npy_intp shape[1]
 *     # check type and assign iou function
 *     if type(dt) == RLEs:             # <<<<<<<<<<<<<<
 *         _iouFun = _rleIou
 *     elif type(dt) == np.ndarray:
 */
    goto __pyx_L7;
  }

  /* "crowdposetools/_mask.pyx":229
 *     if type(dt) == RLEs:
 *         _iouFun = _rleIou
 *     elif type(dt) == np.ndarray:             # <<<<<<<<<<<<<<
 *         _iouFun = _bbIou
 *     else:
 */
  __pyx_t_5 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_dt)), ((PyObject *)__pyx_ptype_5numpy_ndarray), Py_EQ); __Pyx_XGOTREF(__pyx_t_5); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 229, __pyx_L1_error)
  __pyx_t_9 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_9 < 0)) __PYX_ERR(0, 229, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  if (likely(__pyx_t_9)) {

    /* "crowdposetools/_mask.pyx":230
 *         _iouFun = _rleIou
 *     elif type(dt) == np.ndarray:
 *         _iouFun = _bbIou             # <<<<<<<<<<<<<<
 *     else:
 *         raise Exception('input data type not allowed.')
 */
    __Pyx_INCREF(__pyx_v__bbIou);
    __pyx_v__iouFun = __pyx_v__bbIou;

    /* "crowdposetools/_mask.pyx":229
 *     if type(dt) == RLEs:
 *         _iouFun = _rleIou
 *     elif type(dt) == np.ndarray:             # <<<<<<<<<<<<<<
 *         _iouFun = _bbIou
 *     else:
 */
    goto __pyx_L7;
  }

  /* "crowdposetools/_mask.pyx":232
 *         _iouFun = _bbIou
 *     else:
 *         raise Exception('input data type not allowed.')             # <<<<<<<<<<<<<<
 *     _iou = <double*> malloc(m*n* sizeof(double))
 *     iou = np.zeros((m*n, ), dtype=np.double)
 */
  /*else*/ {
    __pyx_t_5 = __Pyx_PyObject_Call(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])), __pyx_tuple__20, NULL); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 232, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_Raise(__pyx_t_5, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __PYX_ERR(0, 232, __pyx_L1_error)
  }
  __pyx_L7:;

  /* "crowdposetools/_mask.pyx":233
 *     else:
 *         raise Exception('input data type not allowed.')
 *     _iou = <double*> malloc(m*n* sizeof(double))             # <<<<<<<<<<<<<<
 *     iou = np.zeros((m*n, ), dtype=np.double)
 *     shape[0] = <np.npy_intp> m*n
 */
  __pyx_v__iou = ((double *)malloc(((__pyx_v_m * __pyx_v_n) * (sizeof(double)))));

  /* "crowdposetools/_mask.pyx":234
 *         raise Exception('input data type not allowed.')
 *     _iou = <double*> malloc(m*n* sizeof(double))
 *     iou = np.zeros((m*n, ), dtype=np.double)             # <<<<<<<<<<<<<<
 *     shape[0] = <np.npy_intp> m*n
 *     iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
 */
  __pyx_t_5 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_t_5, __pyx_n_s_zeros); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = __Pyx_PyInt_From_siz((__pyx_v_m * __pyx_v_n)); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_1 = PyTuple_New(1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_5);
  PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = PyTuple_New(1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_4 = __Pyx_PyObject_GetAttrStr(__pyx_t_2, __pyx_n_s_double); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (PyDict_SetItem(__pyx_t_1, __pyx_n_s_dtype, __pyx_t_4) < 0) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_t_4 = __Pyx_PyObject_Call(__pyx_t_3, __pyx_t_5, __pyx_t_1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 234, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_iou = __pyx_t_4;
  __pyx_t_4 = 0;

  /* "crowdposetools/_mask.pyx":235
 *     _iou = <double*> malloc(m*n* sizeof(double))
 *     iou = np.zeros((m*n, ), dtype=np.double)
 *     shape[0] = <np.npy_intp> m*n             # <<<<<<<<<<<<<<
 *     iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
 *     PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
 */
  (__pyx_v_shape[0]) = (((npy_intp)__pyx_v_m) * __pyx_v_n);

  /* "crowdposetools/_mask.pyx":236
 *     iou = np.zeros((m*n, ), dtype=np.double)
 *     shape[0] = <np.npy_intp> m*n
 *     iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)             # <<<<<<<<<<<<<<
 *     PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
 *     _iouFun(dt, gt, iscrowd, m, n, iou)
 */
  __pyx_t_4 = PyArray_SimpleNewFromData(1, __pyx_v_shape, NPY_DOUBLE, __pyx_v__iou); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 236, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF_SET(__pyx_v_iou, __pyx_t_4);
  __pyx_t_4 = 0;

  /* "crowdposetools/_mask.pyx":237
 *     shape[0] = <np.npy_intp> m*n
 *     iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
 *     PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)             # <<<<<<<<<<<<<<
 *     _iouFun(dt, gt, iscrowd, m, n, iou)
 *     return iou.reshape((m,n), order='F')
 */
  if (!(likely(((__pyx_v_iou) == Py_None) || likely(__Pyx_TypeTest(__pyx_v_iou, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 237, __pyx_L1_error)
  PyArray_ENABLEFLAGS(((PyArrayObject *)__pyx_v_iou), NPY_OWNDATA);

  /* "crowdposetools/_mask.pyx":238
 *     iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
 *     PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
 *     _iouFun(dt, gt, iscrowd, m, n, iou)             # <<<<<<<<<<<<<<
 *     return iou.reshape((m,n), order='F')
 * 
 */
  __pyx_t_1 = __Pyx_PyInt_From_siz(__pyx_v_m); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 238, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_5 = __Pyx_PyInt_From_siz(__pyx_v_n); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 238, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_INCREF(__pyx_v__iouFun);
  __pyx_t_3 = __pyx_v__iouFun; __pyx_t_2 = NULL;
  __pyx_t_10 = 0;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_3))) {
    __pyx_t_2 = PyMethod_GET_SELF(__pyx_t_3);
    if (likely(__pyx_t_2)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_3);
      __Pyx_INCREF(__pyx_t_2);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_3, function);
      __pyx_t_10 = 1;
    }
  }
  #if CYTHON_FAST_PYCALL
  if (PyFunction_Check(__pyx_t_3)) {
    PyObject *__pyx_temp[7] = {__pyx_t_2, __pyx_v_dt, __pyx_v_gt, ((PyObject *)__pyx_v_iscrowd), __pyx_t_1, __pyx_t_5, __pyx_v_iou};
    __pyx_t_4 = __Pyx_PyFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_10, 6+__pyx_t_10); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 238, __pyx_L1_error)
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_GOTREF(__pyx_t_4);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  } else
  #endif
  #if CYTHON_FAST_PYCCALL
  if (__Pyx_PyFastCFunction_Check(__pyx_t_3)) {
    PyObject *__pyx_temp[7] = {__pyx_t_2, __pyx_v_dt, __pyx_v_gt, ((PyObject *)__pyx_v_iscrowd), __pyx_t_1, __pyx_t_5, __pyx_v_iou};
    __pyx_t_4 = __Pyx_PyCFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_10, 6+__pyx_t_10); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 238, __pyx_L1_error)
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_GOTREF(__pyx_t_4);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  } else
  #endif
  {
    __pyx_t_11 = PyTuple_New(6+__pyx_t_10); if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 238, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_11);
    if (__pyx_t_2) {
      __Pyx_GIVEREF(__pyx_t_2); PyTuple_SET_ITEM(__pyx_t_11, 0, __pyx_t_2); __pyx_t_2 = NULL;
    }
    __Pyx_INCREF(__pyx_v_dt);
    __Pyx_GIVEREF(__pyx_v_dt);
    PyTuple_SET_ITEM(__pyx_t_11, 0+__pyx_t_10, __pyx_v_dt);
    __Pyx_INCREF(__pyx_v_gt);
    __Pyx_GIVEREF(__pyx_v_gt);
    PyTuple_SET_ITEM(__pyx_t_11, 1+__pyx_t_10, __pyx_v_gt);
    __Pyx_INCREF(((PyObject *)__pyx_v_iscrowd));
    __Pyx_GIVEREF(((PyObject *)__pyx_v_iscrowd));
    PyTuple_SET_ITEM(__pyx_t_11, 2+__pyx_t_10, ((PyObject *)__pyx_v_iscrowd));
    __Pyx_GIVEREF(__pyx_t_1);
    PyTuple_SET_ITEM(__pyx_t_11, 3+__pyx_t_10, __pyx_t_1);
    __Pyx_GIVEREF(__pyx_t_5);
    PyTuple_SET_ITEM(__pyx_t_11, 4+__pyx_t_10, __pyx_t_5);
    __Pyx_INCREF(__pyx_v_iou);
    __Pyx_GIVEREF(__pyx_v_iou);
    PyTuple_SET_ITEM(__pyx_t_11, 5+__pyx_t_10, __pyx_v_iou);
    __pyx_t_1 = 0;
    __pyx_t_5 = 0;
    __pyx_t_4 = __Pyx_PyObject_Call(__pyx_t_3, __pyx_t_11, NULL); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 238, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __Pyx_DECREF(__pyx_t_11); __pyx_t_11 = 0;
  }
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;

  /* "crowdposetools/_mask.pyx":239
 *     PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
 *     _iouFun(dt, gt, iscrowd, m, n, iou)
 *     return iou.reshape((m,n), order='F')             # <<<<<<<<<<<<<<
 * 
 * def toBbox( rleObjs ):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_4 = __Pyx_PyObject_GetAttrStr(__pyx_v_iou, __pyx_n_s_reshape); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 239, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_3 = __Pyx_PyInt_From_siz(__pyx_v_m); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 239, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_11 = __Pyx_PyInt_From_siz(__pyx_v_n); if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 239, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_11);
  __pyx_t_5 = PyTuple_New(2); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 239, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_GIVEREF(__pyx_t_3);
  PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_3);
  __Pyx_GIVEREF(__pyx_t_11);
  PyTuple_SET_ITEM(__pyx_t_5, 1, __pyx_t_11);
  __pyx_t_3 = 0;
  __pyx_t_11 = 0;
  __pyx_t_11 = PyTuple_New(1); if (unlikely(!__pyx_t_11)) __PYX_ERR(0, 239, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_11);
  __Pyx_GIVEREF(__pyx_t_5);
  PyTuple_SET_ITEM(__pyx_t_11, 0, __pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 239, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_5);
  if (PyDict_SetItem(__pyx_t_5, __pyx_n_s_order, __pyx_n_s_F) < 0) __PYX_ERR(0, 239, __pyx_L1_error)
  __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_4, __pyx_t_11, __pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 239, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __Pyx_DECREF(__pyx_t_11); __pyx_t_11 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_r = __pyx_t_3;
  __pyx_t_3 = 0;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":171
 * 
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):             # <<<<<<<<<<<<<<
 *     def _preproc(objs):
 *         if len(objs) == 0:
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_11);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("crowdposetools._mask.iou", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_iscrowd.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF(__pyx_v__preproc);
  __Pyx_XDECREF(__pyx_v__rleIou);
  __Pyx_XDECREF(__pyx_v__bbIou);
  __Pyx_XDECREF(__pyx_v__len);
  __Pyx_XDECREF((PyObject *)__pyx_v_iscrowd);
  __Pyx_XDECREF(__pyx_v__iouFun);
  __Pyx_XDECREF(__pyx_v_iou);
  __Pyx_XDECREF(__pyx_v_dt);
  __Pyx_XDECREF(__pyx_v_gt);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":241
 *     return iou.reshape((m,n), order='F')
 * 
 * def toBbox( rleObjs ):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef siz n = Rs.n
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_15toBbox(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_15toBbox = {"toBbox", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_15toBbox, METH_O, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_15toBbox(PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("toBbox (wrapper)", 0);
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_14toBbox(__pyx_self, ((PyObject *)__pyx_v_rleObjs));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_14toBbox(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_rleObjs) {
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = 0;
  siz __pyx_v_n;
  BB __pyx_v__bb;
  npy_intp __pyx_v_shape[1];
  PyObject *__pyx_v_bb = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  siz __pyx_t_5;
  PyObject *__pyx_t_6 = NULL;
  __Pyx_RefNannySetupContext("toBbox", 0);

  /* "crowdposetools/_mask.pyx":242
 * 
 * def toBbox( rleObjs ):
 *     cdef RLEs Rs = _frString(rleObjs)             # <<<<<<<<<<<<<<
 *     cdef siz n = Rs.n
 *     cdef BB _bb = <BB> malloc(4*n* sizeof(double))
 */
  __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_frString); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 242, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_3 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_2))) {
    __pyx_t_3 = PyMethod_GET_SELF(__pyx_t_2);
    if (likely(__pyx_t_3)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_2);
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_2, function);
    }
  }
  if (!__pyx_t_3) {
    __pyx_t_1 = __Pyx_PyObject_CallOneArg(__pyx_t_2, __pyx_v_rleObjs); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 242, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 242, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, __pyx_v_rleObjs};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 242, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 242, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_GIVEREF(__pyx_t_3); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_3); __pyx_t_3 = NULL;
      __Pyx_INCREF(__pyx_v_rleObjs);
      __Pyx_GIVEREF(__pyx_v_rleObjs);
      PyTuple_SET_ITEM(__pyx_t_4, 0+1, __pyx_v_rleObjs);
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_4, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 242, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (!(likely(((__pyx_t_1) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_1, __pyx_ptype_14crowdposetools_5_mask_RLEs))))) __PYX_ERR(0, 242, __pyx_L1_error)
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_1);
  __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":243
 * def toBbox( rleObjs ):
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef siz n = Rs.n             # <<<<<<<<<<<<<<
 *     cdef BB _bb = <BB> malloc(4*n* sizeof(double))
 *     rleToBbox( <const RLE*> Rs._R, _bb, n )
 */
  __pyx_t_1 = __Pyx_PyObject_GetAttrStr(((PyObject *)__pyx_v_Rs), __pyx_n_s_n); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 243, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_5 = __Pyx_PyInt_As_siz(__pyx_t_1); if (unlikely((__pyx_t_5 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 243, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_n = __pyx_t_5;

  /* "crowdposetools/_mask.pyx":244
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef siz n = Rs.n
 *     cdef BB _bb = <BB> malloc(4*n* sizeof(double))             # <<<<<<<<<<<<<<
 *     rleToBbox( <const RLE*> Rs._R, _bb, n )
 *     cdef np.npy_intp shape[1]
 */
  __pyx_v__bb = ((BB)malloc(((4 * __pyx_v_n) * (sizeof(double)))));

  /* "crowdposetools/_mask.pyx":245
 *     cdef siz n = Rs.n
 *     cdef BB _bb = <BB> malloc(4*n* sizeof(double))
 *     rleToBbox( <const RLE*> Rs._R, _bb, n )             # <<<<<<<<<<<<<<
 *     cdef np.npy_intp shape[1]
 *     shape[0] = <np.npy_intp> 4*n
 */
  rleToBbox(((RLE const *)__pyx_v_Rs->_R), __pyx_v__bb, __pyx_v_n);

  /* "crowdposetools/_mask.pyx":247
 *     rleToBbox( <const RLE*> Rs._R, _bb, n )
 *     cdef np.npy_intp shape[1]
 *     shape[0] = <np.npy_intp> 4*n             # <<<<<<<<<<<<<<
 *     bb = np.array((1,4*n), dtype=np.double)
 *     bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 4))
 */
  (__pyx_v_shape[0]) = (((npy_intp)4) * __pyx_v_n);

  /* "crowdposetools/_mask.pyx":248
 *     cdef np.npy_intp shape[1]
 *     shape[0] = <np.npy_intp> 4*n
 *     bb = np.array((1,4*n), dtype=np.double)             # <<<<<<<<<<<<<<
 *     bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 4))
 *     PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)
 */
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_array); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_PyInt_From_siz((4 * __pyx_v_n)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_4 = PyTuple_New(2); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_INCREF(__pyx_int_1);
  __Pyx_GIVEREF(__pyx_int_1);
  PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_int_1);
  __Pyx_GIVEREF(__pyx_t_1);
  PyTuple_SET_ITEM(__pyx_t_4, 1, __pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyTuple_New(1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_4);
  PyTuple_SET_ITEM(__pyx_t_1, 0, __pyx_t_4);
  __pyx_t_4 = 0;
  __pyx_t_4 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_6 = __Pyx_PyObject_GetAttrStr(__pyx_t_3, __pyx_n_s_double); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_6);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (PyDict_SetItem(__pyx_t_4, __pyx_n_s_dtype, __pyx_t_6) < 0) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
  __pyx_t_6 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_1, __pyx_t_4); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 248, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_6);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_v_bb = __pyx_t_6;
  __pyx_t_6 = 0;

  /* "crowdposetools/_mask.pyx":249
 *     shape[0] = <np.npy_intp> 4*n
 *     bb = np.array((1,4*n), dtype=np.double)
 *     bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 4))             # <<<<<<<<<<<<<<
 *     PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)
 *     return bb
 */
  __pyx_t_4 = PyArray_SimpleNewFromData(1, __pyx_v_shape, NPY_DOUBLE, __pyx_v__bb); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 249, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_1 = __Pyx_PyObject_GetAttrStr(__pyx_t_4, __pyx_n_s_reshape); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 249, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_t_4 = __Pyx_PyInt_From_siz(__pyx_v_n); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 249, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_2 = PyTuple_New(2); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 249, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_GIVEREF(__pyx_t_4);
  PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_t_4);
  __Pyx_INCREF(__pyx_int_4);
  __Pyx_GIVEREF(__pyx_int_4);
  PyTuple_SET_ITEM(__pyx_t_2, 1, __pyx_int_4);
  __pyx_t_4 = 0;
  __pyx_t_4 = NULL;
  if (CYTHON_UNPACK_METHODS && likely(PyMethod_Check(__pyx_t_1))) {
    __pyx_t_4 = PyMethod_GET_SELF(__pyx_t_1);
    if (likely(__pyx_t_4)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_1);
      __Pyx_INCREF(__pyx_t_4);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_1, function);
    }
  }
  if (!__pyx_t_4) {
    __pyx_t_6 = __Pyx_PyObject_CallOneArg(__pyx_t_1, __pyx_t_2); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 249, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_GOTREF(__pyx_t_6);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[2] = {__pyx_t_4, __pyx_t_2};
      __pyx_t_6 = __Pyx_PyFunction_FastCall(__pyx_t_1, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 249, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[2] = {__pyx_t_4, __pyx_t_2};
      __pyx_t_6 = __Pyx_PyCFunction_FastCall(__pyx_t_1, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 249, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    } else
    #endif
    {
      __pyx_t_3 = PyTuple_New(1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 249, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_GIVEREF(__pyx_t_4); PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_4); __pyx_t_4 = NULL;
      __Pyx_GIVEREF(__pyx_t_2);
      PyTuple_SET_ITEM(__pyx_t_3, 0+1, __pyx_t_2);
      __pyx_t_2 = 0;
      __pyx_t_6 = __Pyx_PyObject_Call(__pyx_t_1, __pyx_t_3, NULL); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 249, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF_SET(__pyx_v_bb, __pyx_t_6);
  __pyx_t_6 = 0;

  /* "crowdposetools/_mask.pyx":250
 *     bb = np.array((1,4*n), dtype=np.double)
 *     bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 4))
 *     PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)             # <<<<<<<<<<<<<<
 *     return bb
 * 
 */
  if (!(likely(((__pyx_v_bb) == Py_None) || likely(__Pyx_TypeTest(__pyx_v_bb, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 250, __pyx_L1_error)
  PyArray_ENABLEFLAGS(((PyArrayObject *)__pyx_v_bb), NPY_OWNDATA);

  /* "crowdposetools/_mask.pyx":251
 *     bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 4))
 *     PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)
 *     return bb             # <<<<<<<<<<<<<<
 * 
 * def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_bb);
  __pyx_r = __pyx_v_bb;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":241
 *     return iou.reshape((m,n), order='F')
 * 
 * def toBbox( rleObjs ):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef siz n = Rs.n
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_AddTraceback("crowdposetools._mask.toBbox", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF(__pyx_v_bb);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":253
 *     return bb
 * 
 * def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef siz n = bb.shape[0]
 *     Rs = RLEs(n)
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_17frBbox(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_17frBbox = {"frBbox", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_17frBbox, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_17frBbox(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyArrayObject *__pyx_v_bb = 0;
  siz __pyx_v_h;
  siz __pyx_v_w;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("frBbox (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_bb,&__pyx_n_s_h,&__pyx_n_s_w,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_bb)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_h)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frBbox", 1, 3, 3, 1); __PYX_ERR(0, 253, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_w)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frBbox", 1, 3, 3, 2); __PYX_ERR(0, 253, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "frBbox") < 0)) __PYX_ERR(0, 253, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
    }
    __pyx_v_bb = ((PyArrayObject *)values[0]);
    __pyx_v_h = __Pyx_PyInt_As_siz(values[1]); if (unlikely((__pyx_v_h == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 253, __pyx_L3_error)
    __pyx_v_w = __Pyx_PyInt_As_siz(values[2]); if (unlikely((__pyx_v_w == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 253, __pyx_L3_error)
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("frBbox", 1, 3, 3, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 253, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.frBbox", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  if (unlikely(!__Pyx_ArgTypeTest(((PyObject *)__pyx_v_bb), __pyx_ptype_5numpy_ndarray, 1, "bb", 0))) __PYX_ERR(0, 253, __pyx_L1_error)
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_16frBbox(__pyx_self, __pyx_v_bb, __pyx_v_h, __pyx_v_w);

  /* function exit code */
  goto __pyx_L0;
  __pyx_L1_error:;
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_16frBbox(CYTHON_UNUSED PyObject *__pyx_self, PyArrayObject *__pyx_v_bb, siz __pyx_v_h, siz __pyx_v_w) {
  siz __pyx_v_n;
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = NULL;
  PyObject *__pyx_v_objs = NULL;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_bb;
  __Pyx_Buffer __pyx_pybuffer_bb;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  __Pyx_RefNannySetupContext("frBbox", 0);
  __pyx_pybuffer_bb.pybuffer.buf = NULL;
  __pyx_pybuffer_bb.refcount = 0;
  __pyx_pybuffernd_bb.data = NULL;
  __pyx_pybuffernd_bb.rcbuffer = &__pyx_pybuffer_bb;
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_bb.rcbuffer->pybuffer, (PyObject*)__pyx_v_bb, &__Pyx_TypeInfo_nn___pyx_t_5numpy_double_t, PyBUF_FORMAT| PyBUF_STRIDES, 2, 0, __pyx_stack) == -1)) __PYX_ERR(0, 253, __pyx_L1_error)
  }
  __pyx_pybuffernd_bb.diminfo[0].strides = __pyx_pybuffernd_bb.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_bb.diminfo[0].shape = __pyx_pybuffernd_bb.rcbuffer->pybuffer.shape[0]; __pyx_pybuffernd_bb.diminfo[1].strides = __pyx_pybuffernd_bb.rcbuffer->pybuffer.strides[1]; __pyx_pybuffernd_bb.diminfo[1].shape = __pyx_pybuffernd_bb.rcbuffer->pybuffer.shape[1];

  /* "crowdposetools/_mask.pyx":254
 * 
 * def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):
 *     cdef siz n = bb.shape[0]             # <<<<<<<<<<<<<<
 *     Rs = RLEs(n)
 *     rleFrBbox( <RLE*> Rs._R, <const BB> bb.data, h, w, n )
 */
  __pyx_v_n = (__pyx_v_bb->dimensions[0]);

  /* "crowdposetools/_mask.pyx":255
 * def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):
 *     cdef siz n = bb.shape[0]
 *     Rs = RLEs(n)             # <<<<<<<<<<<<<<
 *     rleFrBbox( <RLE*> Rs._R, <const BB> bb.data, h, w, n )
 *     objs = _toString(Rs)
 */
  __pyx_t_1 = __Pyx_PyInt_From_siz(__pyx_v_n); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 255, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyObject_CallOneArg(((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), __pyx_t_1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 255, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_2);
  __pyx_t_2 = 0;

  /* "crowdposetools/_mask.pyx":256
 *     cdef siz n = bb.shape[0]
 *     Rs = RLEs(n)
 *     rleFrBbox( <RLE*> Rs._R, <const BB> bb.data, h, w, n )             # <<<<<<<<<<<<<<
 *     objs = _toString(Rs)
 *     return objs
 */
  rleFrBbox(((RLE *)__pyx_v_Rs->_R), ((BB const )__pyx_v_bb->data), __pyx_v_h, __pyx_v_w, __pyx_v_n);

  /* "crowdposetools/_mask.pyx":257
 *     Rs = RLEs(n)
 *     rleFrBbox( <RLE*> Rs._R, <const BB> bb.data, h, w, n )
 *     objs = _toString(Rs)             # <<<<<<<<<<<<<<
 *     return objs
 * 
 */
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_toString); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 257, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_1))) {
    __pyx_t_3 = PyMethod_GET_SELF(__pyx_t_1);
    if (likely(__pyx_t_3)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_1);
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_1, function);
    }
  }
  if (!__pyx_t_3) {
    __pyx_t_2 = __Pyx_PyObject_CallOneArg(__pyx_t_1, ((PyObject *)__pyx_v_Rs)); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 257, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, ((PyObject *)__pyx_v_Rs)};
      __pyx_t_2 = __Pyx_PyFunction_FastCall(__pyx_t_1, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 257, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_2);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[2] = {__pyx_t_3, ((PyObject *)__pyx_v_Rs)};
      __pyx_t_2 = __Pyx_PyCFunction_FastCall(__pyx_t_1, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 257, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
      __Pyx_GOTREF(__pyx_t_2);
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(1+1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 257, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_GIVEREF(__pyx_t_3); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_3); __pyx_t_3 = NULL;
      __Pyx_INCREF(((PyObject *)__pyx_v_Rs));
      __Pyx_GIVEREF(((PyObject *)__pyx_v_Rs));
      PyTuple_SET_ITEM(__pyx_t_4, 0+1, ((PyObject *)__pyx_v_Rs));
      __pyx_t_2 = __Pyx_PyObject_Call(__pyx_t_1, __pyx_t_4, NULL); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 257, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_2);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_v_objs = __pyx_t_2;
  __pyx_t_2 = 0;

  /* "crowdposetools/_mask.pyx":258
 *     rleFrBbox( <RLE*> Rs._R, <const BB> bb.data, h, w, n )
 *     objs = _toString(Rs)
 *     return objs             # <<<<<<<<<<<<<<
 * 
 * def frPoly( poly, siz h, siz w ):
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_objs);
  __pyx_r = __pyx_v_objs;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":253
 *     return bb
 * 
 * def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef siz n = bb.shape[0]
 *     Rs = RLEs(n)
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_bb.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("crowdposetools._mask.frBbox", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_bb.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF(__pyx_v_objs);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":260
 *     return objs
 * 
 * def frPoly( poly, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.double_t, ndim=1] np_poly
 *     n = len(poly)
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_19frPoly(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_19frPoly = {"frPoly", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_19frPoly, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_19frPoly(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_poly = 0;
  siz __pyx_v_h;
  siz __pyx_v_w;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("frPoly (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_poly,&__pyx_n_s_h,&__pyx_n_s_w,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_poly)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_h)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frPoly", 1, 3, 3, 1); __PYX_ERR(0, 260, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_w)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frPoly", 1, 3, 3, 2); __PYX_ERR(0, 260, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "frPoly") < 0)) __PYX_ERR(0, 260, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
    }
    __pyx_v_poly = values[0];
    __pyx_v_h = __Pyx_PyInt_As_siz(values[1]); if (unlikely((__pyx_v_h == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 260, __pyx_L3_error)
    __pyx_v_w = __Pyx_PyInt_As_siz(values[2]); if (unlikely((__pyx_v_w == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 260, __pyx_L3_error)
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("frPoly", 1, 3, 3, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 260, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.frPoly", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_18frPoly(__pyx_self, __pyx_v_poly, __pyx_v_h, __pyx_v_w);

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_18frPoly(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_poly, siz __pyx_v_h, siz __pyx_v_w) {
  PyArrayObject *__pyx_v_np_poly = 0;
  Py_ssize_t __pyx_v_n;
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = NULL;
  PyObject *__pyx_v_i = NULL;
  PyObject *__pyx_v_p = NULL;
  PyObject *__pyx_v_objs = NULL;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_np_poly;
  __Pyx_Buffer __pyx_pybuffer_np_poly;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  Py_ssize_t __pyx_t_1;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *(*__pyx_t_4)(PyObject *);
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  PyObject *__pyx_t_9 = NULL;
  PyArrayObject *__pyx_t_10 = NULL;
  int __pyx_t_11;
  PyObject *__pyx_t_12 = NULL;
  PyObject *__pyx_t_13 = NULL;
  PyObject *__pyx_t_14 = NULL;
  Py_ssize_t __pyx_t_15;
  Py_ssize_t __pyx_t_16;
  __Pyx_RefNannySetupContext("frPoly", 0);
  __pyx_pybuffer_np_poly.pybuffer.buf = NULL;
  __pyx_pybuffer_np_poly.refcount = 0;
  __pyx_pybuffernd_np_poly.data = NULL;
  __pyx_pybuffernd_np_poly.rcbuffer = &__pyx_pybuffer_np_poly;

  /* "crowdposetools/_mask.pyx":262
 * def frPoly( poly, siz h, siz w ):
 *     cdef np.ndarray[np.double_t, ndim=1] np_poly
 *     n = len(poly)             # <<<<<<<<<<<<<<
 *     Rs = RLEs(n)
 *     for i, p in enumerate(poly):
 */
  __pyx_t_1 = PyObject_Length(__pyx_v_poly); if (unlikely(__pyx_t_1 == ((Py_ssize_t)-1))) __PYX_ERR(0, 262, __pyx_L1_error)
  __pyx_v_n = __pyx_t_1;

  /* "crowdposetools/_mask.pyx":263
 *     cdef np.ndarray[np.double_t, ndim=1] np_poly
 *     n = len(poly)
 *     Rs = RLEs(n)             # <<<<<<<<<<<<<<
 *     for i, p in enumerate(poly):
 *         np_poly = np.array(p, dtype=np.double, order='F')
 */
  __pyx_t_2 = PyInt_FromSsize_t(__pyx_v_n); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 263, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_3 = __Pyx_PyObject_CallOneArg(((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), __pyx_t_2); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 263, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_v_Rs = ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_3);
  __pyx_t_3 = 0;

  /* "crowdposetools/_mask.pyx":264
 *     n = len(poly)
 *     Rs = RLEs(n)
 *     for i, p in enumerate(poly):             # <<<<<<<<<<<<<<
 *         np_poly = np.array(p, dtype=np.double, order='F')
 *         rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, int(len(p)/2), h, w )
 */
  __Pyx_INCREF(__pyx_int_0);
  __pyx_t_3 = __pyx_int_0;
  if (likely(PyList_CheckExact(__pyx_v_poly)) || PyTuple_CheckExact(__pyx_v_poly)) {
    __pyx_t_2 = __pyx_v_poly; __Pyx_INCREF(__pyx_t_2); __pyx_t_1 = 0;
    __pyx_t_4 = NULL;
  } else {
    __pyx_t_1 = -1; __pyx_t_2 = PyObject_GetIter(__pyx_v_poly); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 264, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __pyx_t_4 = Py_TYPE(__pyx_t_2)->tp_iternext; if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 264, __pyx_L1_error)
  }
  for (;;) {
    if (likely(!__pyx_t_4)) {
      if (likely(PyList_CheckExact(__pyx_t_2))) {
        if (__pyx_t_1 >= PyList_GET_SIZE(__pyx_t_2)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = PyList_GET_ITEM(__pyx_t_2, __pyx_t_1); __Pyx_INCREF(__pyx_t_5); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 264, __pyx_L1_error)
        #else
        __pyx_t_5 = PySequence_ITEM(__pyx_t_2, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 264, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        #endif
      } else {
        if (__pyx_t_1 >= PyTuple_GET_SIZE(__pyx_t_2)) break;
        #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        __pyx_t_5 = PyTuple_GET_ITEM(__pyx_t_2, __pyx_t_1); __Pyx_INCREF(__pyx_t_5); __pyx_t_1++; if (unlikely(0 < 0)) __PYX_ERR(0, 264, __pyx_L1_error)
        #else
        __pyx_t_5 = PySequence_ITEM(__pyx_t_2, __pyx_t_1); __pyx_t_1++; if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 264, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        #endif
      }
    } else {
      __pyx_t_5 = __pyx_t_4(__pyx_t_2);
      if (unlikely(!__pyx_t_5)) {
        PyObject* exc_type = PyErr_Occurred();
        if (exc_type) {
          if (likely(__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();
          else __PYX_ERR(0, 264, __pyx_L1_error)
        }
        break;
      }
      __Pyx_GOTREF(__pyx_t_5);
    }
    __Pyx_XDECREF_SET(__pyx_v_p, __pyx_t_5);
    __pyx_t_5 = 0;
    __Pyx_INCREF(__pyx_t_3);
    __Pyx_XDECREF_SET(__pyx_v_i, __pyx_t_3);
    __pyx_t_5 = __Pyx_PyInt_AddObjC(__pyx_t_3, __pyx_int_1, 1, 0); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 264, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_3);
    __pyx_t_3 = __pyx_t_5;
    __pyx_t_5 = 0;

    /* "crowdposetools/_mask.pyx":265
 *     Rs = RLEs(n)
 *     for i, p in enumerate(poly):
 *         np_poly = np.array(p, dtype=np.double, order='F')             # <<<<<<<<<<<<<<
 *         rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, int(len(p)/2), h, w )
 *     objs = _toString(Rs)
 */
    __pyx_t_5 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __pyx_t_6 = __Pyx_PyObject_GetAttrStr(__pyx_t_5, __pyx_n_s_array); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_5 = PyTuple_New(1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_INCREF(__pyx_v_p);
    __Pyx_GIVEREF(__pyx_v_p);
    PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_v_p);
    __pyx_t_7 = __Pyx_PyDict_NewPresized(2); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __pyx_t_8 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __pyx_t_9 = __Pyx_PyObject_GetAttrStr(__pyx_t_8, __pyx_n_s_double); if (unlikely(!__pyx_t_9)) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_9);
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    if (PyDict_SetItem(__pyx_t_7, __pyx_n_s_dtype, __pyx_t_9) < 0) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
    if (PyDict_SetItem(__pyx_t_7, __pyx_n_s_order, __pyx_n_s_F) < 0) __PYX_ERR(0, 265, __pyx_L1_error)
    __pyx_t_9 = __Pyx_PyObject_Call(__pyx_t_6, __pyx_t_5, __pyx_t_7); if (unlikely(!__pyx_t_9)) __PYX_ERR(0, 265, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_9);
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    if (!(likely(((__pyx_t_9) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_9, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 265, __pyx_L1_error)
    __pyx_t_10 = ((PyArrayObject *)__pyx_t_9);
    {
      __Pyx_BufFmt_StackElem __pyx_stack[1];
      __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_np_poly.rcbuffer->pybuffer);
      __pyx_t_11 = __Pyx_GetBufferAndValidate(&__pyx_pybuffernd_np_poly.rcbuffer->pybuffer, (PyObject*)__pyx_t_10, &__Pyx_TypeInfo_nn___pyx_t_5numpy_double_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack);
      if (unlikely(__pyx_t_11 < 0)) {
        PyErr_Fetch(&__pyx_t_12, &__pyx_t_13, &__pyx_t_14);
        if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_np_poly.rcbuffer->pybuffer, (PyObject*)__pyx_v_np_poly, &__Pyx_TypeInfo_nn___pyx_t_5numpy_double_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
          Py_XDECREF(__pyx_t_12); Py_XDECREF(__pyx_t_13); Py_XDECREF(__pyx_t_14);
          __Pyx_RaiseBufferFallbackError();
        } else {
          PyErr_Restore(__pyx_t_12, __pyx_t_13, __pyx_t_14);
        }
        __pyx_t_12 = __pyx_t_13 = __pyx_t_14 = 0;
      }
      __pyx_pybuffernd_np_poly.diminfo[0].strides = __pyx_pybuffernd_np_poly.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_np_poly.diminfo[0].shape = __pyx_pybuffernd_np_poly.rcbuffer->pybuffer.shape[0];
      if (unlikely(__pyx_t_11 < 0)) __PYX_ERR(0, 265, __pyx_L1_error)
    }
    __pyx_t_10 = 0;
    __Pyx_XDECREF_SET(__pyx_v_np_poly, ((PyArrayObject *)__pyx_t_9));
    __pyx_t_9 = 0;

    /* "crowdposetools/_mask.pyx":266
 *     for i, p in enumerate(poly):
 *         np_poly = np.array(p, dtype=np.double, order='F')
 *         rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, int(len(p)/2), h, w )             # <<<<<<<<<<<<<<
 *     objs = _toString(Rs)
 *     return objs
 */
    __pyx_t_15 = __Pyx_PyIndex_AsSsize_t(__pyx_v_i); if (unlikely((__pyx_t_15 == (Py_ssize_t)-1) && PyErr_Occurred())) __PYX_ERR(0, 266, __pyx_L1_error)
    __pyx_t_16 = PyObject_Length(__pyx_v_p); if (unlikely(__pyx_t_16 == ((Py_ssize_t)-1))) __PYX_ERR(0, 266, __pyx_L1_error)
    rleFrPoly(((RLE *)(&(__pyx_v_Rs->_R[__pyx_t_15]))), ((double const *)__pyx_v_np_poly->data), ((siz)__Pyx_div_Py_ssize_t(__pyx_t_16, 2)), __pyx_v_h, __pyx_v_w);

    /* "crowdposetools/_mask.pyx":264
 *     n = len(poly)
 *     Rs = RLEs(n)
 *     for i, p in enumerate(poly):             # <<<<<<<<<<<<<<
 *         np_poly = np.array(p, dtype=np.double, order='F')
 *         rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, int(len(p)/2), h, w )
 */
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;

  /* "crowdposetools/_mask.pyx":267
 *         np_poly = np.array(p, dtype=np.double, order='F')
 *         rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, int(len(p)/2), h, w )
 *     objs = _toString(Rs)             # <<<<<<<<<<<<<<
 *     return objs
 * 
 */
  __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_toString); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 267, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_9 = NULL;
  if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_2))) {
    __pyx_t_9 = PyMethod_GET_SELF(__pyx_t_2);
    if (likely(__pyx_t_9)) {
      PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_2);
      __Pyx_INCREF(__pyx_t_9);
      __Pyx_INCREF(function);
      __Pyx_DECREF_SET(__pyx_t_2, function);
    }
  }
  if (!__pyx_t_9) {
    __pyx_t_3 = __Pyx_PyObject_CallOneArg(__pyx_t_2, ((PyObject *)__pyx_v_Rs)); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 267, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
  } else {
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_9, ((PyObject *)__pyx_v_Rs)};
      __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 267, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_9); __pyx_t_9 = 0;
      __Pyx_GOTREF(__pyx_t_3);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_2)) {
      PyObject *__pyx_temp[2] = {__pyx_t_9, ((PyObject *)__pyx_v_Rs)};
      __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_2, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 267, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_9); __pyx_t_9 = 0;
      __Pyx_GOTREF(__pyx_t_3);
    } else
    #endif
    {
      __pyx_t_7 = PyTuple_New(1+1); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 267, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_7);
      __Pyx_GIVEREF(__pyx_t_9); PyTuple_SET_ITEM(__pyx_t_7, 0, __pyx_t_9); __pyx_t_9 = NULL;
      __Pyx_INCREF(((PyObject *)__pyx_v_Rs));
      __Pyx_GIVEREF(((PyObject *)__pyx_v_Rs));
      PyTuple_SET_ITEM(__pyx_t_7, 0+1, ((PyObject *)__pyx_v_Rs));
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_2, __pyx_t_7, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 267, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    }
  }
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_v_objs = __pyx_t_3;
  __pyx_t_3 = 0;

  /* "crowdposetools/_mask.pyx":268
 *         rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, int(len(p)/2), h, w )
 *     objs = _toString(Rs)
 *     return objs             # <<<<<<<<<<<<<<
 * 
 * def frUncompressedRLE(ucRles, siz h, siz w):
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_objs);
  __pyx_r = __pyx_v_objs;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":260
 *     return objs
 * 
 * def frPoly( poly, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.double_t, ndim=1] np_poly
 *     n = len(poly)
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_XDECREF(__pyx_t_9);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_np_poly.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("crowdposetools._mask.frPoly", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_np_poly.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_np_poly);
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XDECREF(__pyx_v_i);
  __Pyx_XDECREF(__pyx_v_p);
  __Pyx_XDECREF(__pyx_v_objs);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":270
 *     return objs
 * 
 * def frUncompressedRLE(ucRles, siz h, siz w):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.uint32_t, ndim=1] cnts
 *     cdef RLE R
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_21frUncompressedRLE(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_21frUncompressedRLE = {"frUncompressedRLE", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_21frUncompressedRLE, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_21frUncompressedRLE(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_ucRles = 0;
  CYTHON_UNUSED siz __pyx_v_h;
  CYTHON_UNUSED siz __pyx_v_w;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("frUncompressedRLE (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_ucRles,&__pyx_n_s_h,&__pyx_n_s_w,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_ucRles)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_h)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frUncompressedRLE", 1, 3, 3, 1); __PYX_ERR(0, 270, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_w)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frUncompressedRLE", 1, 3, 3, 2); __PYX_ERR(0, 270, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "frUncompressedRLE") < 0)) __PYX_ERR(0, 270, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
    }
    __pyx_v_ucRles = values[0];
    __pyx_v_h = __Pyx_PyInt_As_siz(values[1]); if (unlikely((__pyx_v_h == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 270, __pyx_L3_error)
    __pyx_v_w = __Pyx_PyInt_As_siz(values[2]); if (unlikely((__pyx_v_w == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 270, __pyx_L3_error)
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("frUncompressedRLE", 1, 3, 3, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 270, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.frUncompressedRLE", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_20frUncompressedRLE(__pyx_self, __pyx_v_ucRles, __pyx_v_h, __pyx_v_w);

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_20frUncompressedRLE(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_ucRles, CYTHON_UNUSED siz __pyx_v_h, CYTHON_UNUSED siz __pyx_v_w) {
  PyArrayObject *__pyx_v_cnts = 0;
  RLE __pyx_v_R;
  uint *__pyx_v_data;
  Py_ssize_t __pyx_v_n;
  PyObject *__pyx_v_objs = NULL;
  Py_ssize_t __pyx_v_i;
  struct __pyx_obj_14crowdposetools_5_mask_RLEs *__pyx_v_Rs = NULL;
  Py_ssize_t __pyx_v_j;
  __Pyx_LocalBuf_ND __pyx_pybuffernd_cnts;
  __Pyx_Buffer __pyx_pybuffer_cnts;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  Py_ssize_t __pyx_t_1;
  PyObject *__pyx_t_2 = NULL;
  Py_ssize_t __pyx_t_3;
  Py_ssize_t __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  PyArrayObject *__pyx_t_9 = NULL;
  int __pyx_t_10;
  PyObject *__pyx_t_11 = NULL;
  PyObject *__pyx_t_12 = NULL;
  PyObject *__pyx_t_13 = NULL;
  Py_ssize_t __pyx_t_14;
  Py_ssize_t __pyx_t_15;
  Py_ssize_t __pyx_t_16;
  Py_ssize_t __pyx_t_17;
  RLE __pyx_t_18;
  siz __pyx_t_19;
  int __pyx_t_20;
  __Pyx_RefNannySetupContext("frUncompressedRLE", 0);
  __pyx_pybuffer_cnts.pybuffer.buf = NULL;
  __pyx_pybuffer_cnts.refcount = 0;
  __pyx_pybuffernd_cnts.data = NULL;
  __pyx_pybuffernd_cnts.rcbuffer = &__pyx_pybuffer_cnts;

  /* "crowdposetools/_mask.pyx":274
 *     cdef RLE R
 *     cdef uint *data
 *     n = len(ucRles)             # <<<<<<<<<<<<<<
 *     objs = []
 *     for i in range(n):
 */
  __pyx_t_1 = PyObject_Length(__pyx_v_ucRles); if (unlikely(__pyx_t_1 == ((Py_ssize_t)-1))) __PYX_ERR(0, 274, __pyx_L1_error)
  __pyx_v_n = __pyx_t_1;

  /* "crowdposetools/_mask.pyx":275
 *     cdef uint *data
 *     n = len(ucRles)
 *     objs = []             # <<<<<<<<<<<<<<
 *     for i in range(n):
 *         Rs = RLEs(1)
 */
  __pyx_t_2 = PyList_New(0); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 275, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_v_objs = ((PyObject*)__pyx_t_2);
  __pyx_t_2 = 0;

  /* "crowdposetools/_mask.pyx":276
 *     n = len(ucRles)
 *     objs = []
 *     for i in range(n):             # <<<<<<<<<<<<<<
 *         Rs = RLEs(1)
 *         cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)
 */
  __pyx_t_1 = __pyx_v_n;
  __pyx_t_3 = __pyx_t_1;
  for (__pyx_t_4 = 0; __pyx_t_4 < __pyx_t_3; __pyx_t_4+=1) {
    __pyx_v_i = __pyx_t_4;

    /* "crowdposetools/_mask.pyx":277
 *     objs = []
 *     for i in range(n):
 *         Rs = RLEs(1)             # <<<<<<<<<<<<<<
 *         cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)
 *         # time for malloc can be saved here but it's fine
 */
    __pyx_t_2 = __Pyx_PyObject_Call(((PyObject *)__pyx_ptype_14crowdposetools_5_mask_RLEs), __pyx_tuple__21, NULL); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 277, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_XDECREF_SET(__pyx_v_Rs, ((struct __pyx_obj_14crowdposetools_5_mask_RLEs *)__pyx_t_2));
    __pyx_t_2 = 0;

    /* "crowdposetools/_mask.pyx":278
 *     for i in range(n):
 *         Rs = RLEs(1)
 *         cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)             # <<<<<<<<<<<<<<
 *         # time for malloc can be saved here but it's fine
 *         data = <uint*> malloc(len(cnts)* sizeof(uint))
 */
    __pyx_t_2 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __pyx_t_5 = __Pyx_PyObject_GetAttrStr(__pyx_t_2, __pyx_n_s_array); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __pyx_t_2 = __Pyx_GetItemInt(__pyx_v_ucRles, __pyx_v_i, Py_ssize_t, 1, PyInt_FromSsize_t, 0, 1, 1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __pyx_t_6 = __Pyx_PyObject_Dict_GetItem(__pyx_t_2, __pyx_n_s_counts); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __pyx_t_2 = PyTuple_New(1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_GIVEREF(__pyx_t_6);
    PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_t_6);
    __pyx_t_6 = 0;
    __pyx_t_6 = __Pyx_PyDict_NewPresized(1); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __pyx_t_7 = __Pyx_GetModuleGlobalName(__pyx_n_s_np); if (unlikely(!__pyx_t_7)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_7);
    __pyx_t_8 = __Pyx_PyObject_GetAttrStr(__pyx_t_7, __pyx_n_s_uint32); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
    if (PyDict_SetItem(__pyx_t_6, __pyx_n_s_dtype, __pyx_t_8) < 0) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_8 = __Pyx_PyObject_Call(__pyx_t_5, __pyx_t_2, __pyx_t_6); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 278, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    if (!(likely(((__pyx_t_8) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_8, __pyx_ptype_5numpy_ndarray))))) __PYX_ERR(0, 278, __pyx_L1_error)
    __pyx_t_9 = ((PyArrayObject *)__pyx_t_8);
    {
      __Pyx_BufFmt_StackElem __pyx_stack[1];
      __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_cnts.rcbuffer->pybuffer);
      __pyx_t_10 = __Pyx_GetBufferAndValidate(&__pyx_pybuffernd_cnts.rcbuffer->pybuffer, (PyObject*)__pyx_t_9, &__Pyx_TypeInfo_nn___pyx_t_5numpy_uint32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack);
      if (unlikely(__pyx_t_10 < 0)) {
        PyErr_Fetch(&__pyx_t_11, &__pyx_t_12, &__pyx_t_13);
        if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_pybuffernd_cnts.rcbuffer->pybuffer, (PyObject*)__pyx_v_cnts, &__Pyx_TypeInfo_nn___pyx_t_5numpy_uint32_t, PyBUF_FORMAT| PyBUF_STRIDES, 1, 0, __pyx_stack) == -1)) {
          Py_XDECREF(__pyx_t_11); Py_XDECREF(__pyx_t_12); Py_XDECREF(__pyx_t_13);
          __Pyx_RaiseBufferFallbackError();
        } else {
          PyErr_Restore(__pyx_t_11, __pyx_t_12, __pyx_t_13);
        }
        __pyx_t_11 = __pyx_t_12 = __pyx_t_13 = 0;
      }
      __pyx_pybuffernd_cnts.diminfo[0].strides = __pyx_pybuffernd_cnts.rcbuffer->pybuffer.strides[0]; __pyx_pybuffernd_cnts.diminfo[0].shape = __pyx_pybuffernd_cnts.rcbuffer->pybuffer.shape[0];
      if (unlikely(__pyx_t_10 < 0)) __PYX_ERR(0, 278, __pyx_L1_error)
    }
    __pyx_t_9 = 0;
    __Pyx_XDECREF_SET(__pyx_v_cnts, ((PyArrayObject *)__pyx_t_8));
    __pyx_t_8 = 0;

    /* "crowdposetools/_mask.pyx":280
 *         cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)
 *         # time for malloc can be saved here but it's fine
 *         data = <uint*> malloc(len(cnts)* sizeof(uint))             # <<<<<<<<<<<<<<
 *         for j in range(len(cnts)):
 *             data[j] = <uint> cnts[j]
 */
    __pyx_t_14 = PyObject_Length(((PyObject *)__pyx_v_cnts)); if (unlikely(__pyx_t_14 == ((Py_ssize_t)-1))) __PYX_ERR(0, 280, __pyx_L1_error)
    __pyx_v_data = ((uint *)malloc((__pyx_t_14 * (sizeof(unsigned int)))));

    /* "crowdposetools/_mask.pyx":281
 *         # time for malloc can be saved here but it's fine
 *         data = <uint*> malloc(len(cnts)* sizeof(uint))
 *         for j in range(len(cnts)):             # <<<<<<<<<<<<<<
 *             data[j] = <uint> cnts[j]
 *         R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)
 */
    __pyx_t_14 = PyObject_Length(((PyObject *)__pyx_v_cnts)); if (unlikely(__pyx_t_14 == ((Py_ssize_t)-1))) __PYX_ERR(0, 281, __pyx_L1_error)
    __pyx_t_15 = __pyx_t_14;
    for (__pyx_t_16 = 0; __pyx_t_16 < __pyx_t_15; __pyx_t_16+=1) {
      __pyx_v_j = __pyx_t_16;

      /* "crowdposetools/_mask.pyx":282
 *         data = <uint*> malloc(len(cnts)* sizeof(uint))
 *         for j in range(len(cnts)):
 *             data[j] = <uint> cnts[j]             # <<<<<<<<<<<<<<
 *         R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)
 *         Rs._R[0] = R
 */
      __pyx_t_17 = __pyx_v_j;
      __pyx_t_10 = -1;
      if (__pyx_t_17 < 0) {
        __pyx_t_17 += __pyx_pybuffernd_cnts.diminfo[0].shape;
        if (unlikely(__pyx_t_17 < 0)) __pyx_t_10 = 0;
      } else if (unlikely(__pyx_t_17 >= __pyx_pybuffernd_cnts.diminfo[0].shape)) __pyx_t_10 = 0;
      if (unlikely(__pyx_t_10 != -1)) {
        __Pyx_RaiseBufferIndexError(__pyx_t_10);
        __PYX_ERR(0, 282, __pyx_L1_error)
      }
      (__pyx_v_data[__pyx_v_j]) = ((uint)(*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_uint32_t *, __pyx_pybuffernd_cnts.rcbuffer->pybuffer.buf, __pyx_t_17, __pyx_pybuffernd_cnts.diminfo[0].strides)));
    }

    /* "crowdposetools/_mask.pyx":283
 *         for j in range(len(cnts)):
 *             data[j] = <uint> cnts[j]
 *         R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)             # <<<<<<<<<<<<<<
 *         Rs._R[0] = R
 *         objs.append(_toString(Rs)[0])
 */
    __pyx_t_8 = __Pyx_GetItemInt(__pyx_v_ucRles, __pyx_v_i, Py_ssize_t, 1, PyInt_FromSsize_t, 0, 1, 1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __pyx_t_6 = __Pyx_PyObject_Dict_GetItem(__pyx_t_8, __pyx_n_s_size); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_8 = __Pyx_GetItemInt(__pyx_t_6, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_19 = __Pyx_PyInt_As_siz(__pyx_t_8); if (unlikely((__pyx_t_19 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_18.h = __pyx_t_19;
    __pyx_t_8 = __Pyx_GetItemInt(__pyx_v_ucRles, __pyx_v_i, Py_ssize_t, 1, PyInt_FromSsize_t, 0, 1, 1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __pyx_t_6 = __Pyx_PyObject_Dict_GetItem(__pyx_t_8, __pyx_n_s_size); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_8 = __Pyx_GetItemInt(__pyx_t_6, 1, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_8);
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_19 = __Pyx_PyInt_As_siz(__pyx_t_8); if (unlikely((__pyx_t_19 == ((siz)-1)) && PyErr_Occurred())) __PYX_ERR(0, 283, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_18.w = __pyx_t_19;
    __pyx_t_14 = PyObject_Length(((PyObject *)__pyx_v_cnts)); if (unlikely(__pyx_t_14 == ((Py_ssize_t)-1))) __PYX_ERR(0, 283, __pyx_L1_error)
    __pyx_t_18.m = __pyx_t_14;
    __pyx_t_18.cnts = ((uint *)__pyx_v_data);
    __pyx_v_R = __pyx_t_18;

    /* "crowdposetools/_mask.pyx":284
 *             data[j] = <uint> cnts[j]
 *         R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)
 *         Rs._R[0] = R             # <<<<<<<<<<<<<<
 *         objs.append(_toString(Rs)[0])
 *     return objs
 */
    (__pyx_v_Rs->_R[0]) = __pyx_v_R;

    /* "crowdposetools/_mask.pyx":285
 *         R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)
 *         Rs._R[0] = R
 *         objs.append(_toString(Rs)[0])             # <<<<<<<<<<<<<<
 *     return objs
 * 
 */
    __pyx_t_6 = __Pyx_GetModuleGlobalName(__pyx_n_s_toString); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 285, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __pyx_t_2 = NULL;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_6))) {
      __pyx_t_2 = PyMethod_GET_SELF(__pyx_t_6);
      if (likely(__pyx_t_2)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_6);
        __Pyx_INCREF(__pyx_t_2);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_6, function);
      }
    }
    if (!__pyx_t_2) {
      __pyx_t_8 = __Pyx_PyObject_CallOneArg(__pyx_t_6, ((PyObject *)__pyx_v_Rs)); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 285, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_8);
    } else {
      #if CYTHON_FAST_PYCALL
      if (PyFunction_Check(__pyx_t_6)) {
        PyObject *__pyx_temp[2] = {__pyx_t_2, ((PyObject *)__pyx_v_Rs)};
        __pyx_t_8 = __Pyx_PyFunction_FastCall(__pyx_t_6, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 285, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
        __Pyx_GOTREF(__pyx_t_8);
      } else
      #endif
      #if CYTHON_FAST_PYCCALL
      if (__Pyx_PyFastCFunction_Check(__pyx_t_6)) {
        PyObject *__pyx_temp[2] = {__pyx_t_2, ((PyObject *)__pyx_v_Rs)};
        __pyx_t_8 = __Pyx_PyCFunction_FastCall(__pyx_t_6, __pyx_temp+1-1, 1+1); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 285, __pyx_L1_error)
        __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
        __Pyx_GOTREF(__pyx_t_8);
      } else
      #endif
      {
        __pyx_t_5 = PyTuple_New(1+1); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 285, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_5);
        __Pyx_GIVEREF(__pyx_t_2); PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_2); __pyx_t_2 = NULL;
        __Pyx_INCREF(((PyObject *)__pyx_v_Rs));
        __Pyx_GIVEREF(((PyObject *)__pyx_v_Rs));
        PyTuple_SET_ITEM(__pyx_t_5, 0+1, ((PyObject *)__pyx_v_Rs));
        __pyx_t_8 = __Pyx_PyObject_Call(__pyx_t_6, __pyx_t_5, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(0, 285, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_8);
        __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      }
    }
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    __pyx_t_6 = __Pyx_GetItemInt(__pyx_t_8, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 285, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
    __pyx_t_20 = __Pyx_PyList_Append(__pyx_v_objs, __pyx_t_6); if (unlikely(__pyx_t_20 == ((int)-1))) __PYX_ERR(0, 285, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
  }

  /* "crowdposetools/_mask.pyx":286
 *         Rs._R[0] = R
 *         objs.append(_toString(Rs)[0])
 *     return objs             # <<<<<<<<<<<<<<
 * 
 * def frPyObjects(pyobj, h, w):
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_objs);
  __pyx_r = __pyx_v_objs;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":270
 *     return objs
 * 
 * def frUncompressedRLE(ucRles, siz h, siz w):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.uint32_t, ndim=1] cnts
 *     cdef RLE R
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_cnts.rcbuffer->pybuffer);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("crowdposetools._mask.frUncompressedRLE", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_pybuffernd_cnts.rcbuffer->pybuffer);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_cnts);
  __Pyx_XDECREF(__pyx_v_objs);
  __Pyx_XDECREF((PyObject *)__pyx_v_Rs);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "crowdposetools/_mask.pyx":288
 *     return objs
 * 
 * def frPyObjects(pyobj, h, w):             # <<<<<<<<<<<<<<
 *     # encode rle from a list of python objects
 *     if type(pyobj) == np.ndarray:
 */

/* Python wrapper */
static PyObject *__pyx_pw_14crowdposetools_5_mask_23frPyObjects(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyMethodDef __pyx_mdef_14crowdposetools_5_mask_23frPyObjects = {"frPyObjects", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_23frPyObjects, METH_VARARGS|METH_KEYWORDS, 0};
static PyObject *__pyx_pw_14crowdposetools_5_mask_23frPyObjects(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_pyobj = 0;
  PyObject *__pyx_v_h = 0;
  PyObject *__pyx_v_w = 0;
  PyObject *__pyx_r = 0;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("frPyObjects (wrapper)", 0);
  {
    static PyObject **__pyx_pyargnames[] = {&__pyx_n_s_pyobj,&__pyx_n_s_h,&__pyx_n_s_w,0};
    PyObject* values[3] = {0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      const Py_ssize_t pos_args = PyTuple_GET_SIZE(__pyx_args);
      switch (pos_args) {
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        CYTHON_FALLTHROUGH;
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        CYTHON_FALLTHROUGH;
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        CYTHON_FALLTHROUGH;
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (pos_args) {
        case  0:
        if (likely((values[0] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_pyobj)) != 0)) kw_args--;
        else goto __pyx_L5_argtuple_error;
        CYTHON_FALLTHROUGH;
        case  1:
        if (likely((values[1] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_h)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frPyObjects", 1, 3, 3, 1); __PYX_ERR(0, 288, __pyx_L3_error)
        }
        CYTHON_FALLTHROUGH;
        case  2:
        if (likely((values[2] = __Pyx_PyDict_GetItemStr(__pyx_kwds, __pyx_n_s_w)) != 0)) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("frPyObjects", 1, 3, 3, 2); __PYX_ERR(0, 288, __pyx_L3_error)
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, pos_args, "frPyObjects") < 0)) __PYX_ERR(0, 288, __pyx_L3_error)
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 3) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
    }
    __pyx_v_pyobj = values[0];
    __pyx_v_h = values[1];
    __pyx_v_w = values[2];
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("frPyObjects", 1, 3, 3, PyTuple_GET_SIZE(__pyx_args)); __PYX_ERR(0, 288, __pyx_L3_error)
  __pyx_L3_error:;
  __Pyx_AddTraceback("crowdposetools._mask.frPyObjects", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __pyx_r = __pyx_pf_14crowdposetools_5_mask_22frPyObjects(__pyx_self, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w);

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_pf_14crowdposetools_5_mask_22frPyObjects(CYTHON_UNUSED PyObject *__pyx_self, PyObject *__pyx_v_pyobj, PyObject *__pyx_v_h, PyObject *__pyx_v_w) {
  PyObject *__pyx_v_objs = NULL;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  int __pyx_t_5;
  PyObject *__pyx_t_6 = NULL;
  int __pyx_t_7;
  Py_ssize_t __pyx_t_8;
  int __pyx_t_9;
  PyObject *__pyx_t_10 = NULL;
  __Pyx_RefNannySetupContext("frPyObjects", 0);

  /* "crowdposetools/_mask.pyx":290
 * def frPyObjects(pyobj, h, w):
 *     # encode rle from a list of python objects
 *     if type(pyobj) == np.ndarray:             # <<<<<<<<<<<<<<
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:
 */
  __pyx_t_1 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_pyobj)), ((PyObject *)__pyx_ptype_5numpy_ndarray), Py_EQ); __Pyx_XGOTREF(__pyx_t_1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 290, __pyx_L1_error)
  __pyx_t_2 = __Pyx_PyObject_IsTrue(__pyx_t_1); if (unlikely(__pyx_t_2 < 0)) __PYX_ERR(0, 290, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":291
 *     # encode rle from a list of python objects
 *     if type(pyobj) == np.ndarray:
 *         objs = frBbox(pyobj, h, w)             # <<<<<<<<<<<<<<
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:
 *         objs = frBbox(pyobj, h, w)
 */
    __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_frBbox); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 291, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_4 = NULL;
    __pyx_t_5 = 0;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_3))) {
      __pyx_t_4 = PyMethod_GET_SELF(__pyx_t_3);
      if (likely(__pyx_t_4)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_3);
        __Pyx_INCREF(__pyx_t_4);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_3, function);
        __pyx_t_5 = 1;
      }
    }
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_4, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 291, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_4, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 291, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_6 = PyTuple_New(3+__pyx_t_5); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 291, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      if (__pyx_t_4) {
        __Pyx_GIVEREF(__pyx_t_4); PyTuple_SET_ITEM(__pyx_t_6, 0, __pyx_t_4); __pyx_t_4 = NULL;
      }
      __Pyx_INCREF(__pyx_v_pyobj);
      __Pyx_GIVEREF(__pyx_v_pyobj);
      PyTuple_SET_ITEM(__pyx_t_6, 0+__pyx_t_5, __pyx_v_pyobj);
      __Pyx_INCREF(__pyx_v_h);
      __Pyx_GIVEREF(__pyx_v_h);
      PyTuple_SET_ITEM(__pyx_t_6, 1+__pyx_t_5, __pyx_v_h);
      __Pyx_INCREF(__pyx_v_w);
      __Pyx_GIVEREF(__pyx_v_w);
      PyTuple_SET_ITEM(__pyx_t_6, 2+__pyx_t_5, __pyx_v_w);
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_3, __pyx_t_6, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 291, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    }
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_v_objs = __pyx_t_1;
    __pyx_t_1 = 0;

    /* "crowdposetools/_mask.pyx":290
 * def frPyObjects(pyobj, h, w):
 *     # encode rle from a list of python objects
 *     if type(pyobj) == np.ndarray:             # <<<<<<<<<<<<<<
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":292
 *     if type(pyobj) == np.ndarray:
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:             # <<<<<<<<<<<<<<
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 */
  __pyx_t_1 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_pyobj)), ((PyObject *)(&PyList_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 292, __pyx_L1_error)
  __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_1); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 292, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (__pyx_t_7) {
  } else {
    __pyx_t_2 = __pyx_t_7;
    goto __pyx_L4_bool_binop_done;
  }
  __pyx_t_1 = __Pyx_GetItemInt(__pyx_v_pyobj, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 292, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_8 = PyObject_Length(__pyx_t_1); if (unlikely(__pyx_t_8 == ((Py_ssize_t)-1))) __PYX_ERR(0, 292, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_7 = ((__pyx_t_8 == 4) != 0);
  __pyx_t_2 = __pyx_t_7;
  __pyx_L4_bool_binop_done:;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":293
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:
 *         objs = frBbox(pyobj, h, w)             # <<<<<<<<<<<<<<
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 *         objs = frPoly(pyobj, h, w)
 */
    __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_frBbox); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 293, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_6 = NULL;
    __pyx_t_5 = 0;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_3))) {
      __pyx_t_6 = PyMethod_GET_SELF(__pyx_t_3);
      if (likely(__pyx_t_6)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_3);
        __Pyx_INCREF(__pyx_t_6);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_3, function);
        __pyx_t_5 = 1;
      }
    }
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_6, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 293, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_6, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 293, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(3+__pyx_t_5); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 293, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      if (__pyx_t_6) {
        __Pyx_GIVEREF(__pyx_t_6); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_6); __pyx_t_6 = NULL;
      }
      __Pyx_INCREF(__pyx_v_pyobj);
      __Pyx_GIVEREF(__pyx_v_pyobj);
      PyTuple_SET_ITEM(__pyx_t_4, 0+__pyx_t_5, __pyx_v_pyobj);
      __Pyx_INCREF(__pyx_v_h);
      __Pyx_GIVEREF(__pyx_v_h);
      PyTuple_SET_ITEM(__pyx_t_4, 1+__pyx_t_5, __pyx_v_h);
      __Pyx_INCREF(__pyx_v_w);
      __Pyx_GIVEREF(__pyx_v_w);
      PyTuple_SET_ITEM(__pyx_t_4, 2+__pyx_t_5, __pyx_v_w);
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_3, __pyx_t_4, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 293, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_v_objs = __pyx_t_1;
    __pyx_t_1 = 0;

    /* "crowdposetools/_mask.pyx":292
 *     if type(pyobj) == np.ndarray:
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:             # <<<<<<<<<<<<<<
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":294
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:             # <<<<<<<<<<<<<<
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \
 */
  __pyx_t_1 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_pyobj)), ((PyObject *)(&PyList_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 294, __pyx_L1_error)
  __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_1); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 294, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (__pyx_t_7) {
  } else {
    __pyx_t_2 = __pyx_t_7;
    goto __pyx_L6_bool_binop_done;
  }
  __pyx_t_1 = __Pyx_GetItemInt(__pyx_v_pyobj, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 294, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_8 = PyObject_Length(__pyx_t_1); if (unlikely(__pyx_t_8 == ((Py_ssize_t)-1))) __PYX_ERR(0, 294, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_7 = ((__pyx_t_8 > 4) != 0);
  __pyx_t_2 = __pyx_t_7;
  __pyx_L6_bool_binop_done:;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":295
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 *         objs = frPoly(pyobj, h, w)             # <<<<<<<<<<<<<<
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
 */
    __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_frPoly); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 295, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_4 = NULL;
    __pyx_t_5 = 0;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_3))) {
      __pyx_t_4 = PyMethod_GET_SELF(__pyx_t_3);
      if (likely(__pyx_t_4)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_3);
        __Pyx_INCREF(__pyx_t_4);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_3, function);
        __pyx_t_5 = 1;
      }
    }
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_4, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 295, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_4, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 295, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
    } else
    #endif
    {
      __pyx_t_6 = PyTuple_New(3+__pyx_t_5); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 295, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      if (__pyx_t_4) {
        __Pyx_GIVEREF(__pyx_t_4); PyTuple_SET_ITEM(__pyx_t_6, 0, __pyx_t_4); __pyx_t_4 = NULL;
      }
      __Pyx_INCREF(__pyx_v_pyobj);
      __Pyx_GIVEREF(__pyx_v_pyobj);
      PyTuple_SET_ITEM(__pyx_t_6, 0+__pyx_t_5, __pyx_v_pyobj);
      __Pyx_INCREF(__pyx_v_h);
      __Pyx_GIVEREF(__pyx_v_h);
      PyTuple_SET_ITEM(__pyx_t_6, 1+__pyx_t_5, __pyx_v_h);
      __Pyx_INCREF(__pyx_v_w);
      __Pyx_GIVEREF(__pyx_v_w);
      PyTuple_SET_ITEM(__pyx_t_6, 2+__pyx_t_5, __pyx_v_w);
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_3, __pyx_t_6, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 295, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    }
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_v_objs = __pyx_t_1;
    __pyx_t_1 = 0;

    /* "crowdposetools/_mask.pyx":294
 *     elif type(pyobj) == list and len(pyobj[0]) == 4:
 *         objs = frBbox(pyobj, h, w)
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:             # <<<<<<<<<<<<<<
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":296
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \             # <<<<<<<<<<<<<<
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
 *         objs = frUncompressedRLE(pyobj, h, w)
 */
  __pyx_t_1 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_pyobj)), ((PyObject *)(&PyList_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 296, __pyx_L1_error)
  __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_1); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 296, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (__pyx_t_7) {
  } else {
    __pyx_t_2 = __pyx_t_7;
    goto __pyx_L8_bool_binop_done;
  }

  /* "crowdposetools/_mask.pyx":297
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:             # <<<<<<<<<<<<<<
 *         objs = frUncompressedRLE(pyobj, h, w)
 *     # encode rle from single python object
 */
  __pyx_t_1 = __Pyx_GetItemInt(__pyx_v_pyobj, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 296, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);

  /* "crowdposetools/_mask.pyx":296
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \             # <<<<<<<<<<<<<<
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
 *         objs = frUncompressedRLE(pyobj, h, w)
 */
  __pyx_t_3 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_t_1)), ((PyObject *)(&PyDict_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 296, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 296, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (__pyx_t_7) {
  } else {
    __pyx_t_2 = __pyx_t_7;
    goto __pyx_L8_bool_binop_done;
  }

  /* "crowdposetools/_mask.pyx":297
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:             # <<<<<<<<<<<<<<
 *         objs = frUncompressedRLE(pyobj, h, w)
 *     # encode rle from single python object
 */
  __pyx_t_3 = __Pyx_GetItemInt(__pyx_v_pyobj, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 297, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_7 = (__Pyx_PySequence_ContainsTF(__pyx_n_s_counts, __pyx_t_3, Py_EQ)); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 297, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_9 = (__pyx_t_7 != 0);
  if (__pyx_t_9) {
  } else {
    __pyx_t_2 = __pyx_t_9;
    goto __pyx_L8_bool_binop_done;
  }
  __pyx_t_3 = __Pyx_GetItemInt(__pyx_v_pyobj, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 297, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_9 = (__Pyx_PySequence_ContainsTF(__pyx_n_s_size, __pyx_t_3, Py_EQ)); if (unlikely(__pyx_t_9 < 0)) __PYX_ERR(0, 297, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __pyx_t_7 = (__pyx_t_9 != 0);
  __pyx_t_2 = __pyx_t_7;
  __pyx_L8_bool_binop_done:;

  /* "crowdposetools/_mask.pyx":296
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \             # <<<<<<<<<<<<<<
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
 *         objs = frUncompressedRLE(pyobj, h, w)
 */
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":298
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
 *         objs = frUncompressedRLE(pyobj, h, w)             # <<<<<<<<<<<<<<
 *     # encode rle from single python object
 *     elif type(pyobj) == list and len(pyobj) == 4:
 */
    __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_frUncompressedRLE); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 298, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_6 = NULL;
    __pyx_t_5 = 0;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_1))) {
      __pyx_t_6 = PyMethod_GET_SELF(__pyx_t_1);
      if (likely(__pyx_t_6)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_1);
        __Pyx_INCREF(__pyx_t_6);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_1, function);
        __pyx_t_5 = 1;
      }
    }
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[4] = {__pyx_t_6, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_1, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 298, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_3);
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[4] = {__pyx_t_6, __pyx_v_pyobj, __pyx_v_h, __pyx_v_w};
      __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_1, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 298, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_3);
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(3+__pyx_t_5); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 298, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      if (__pyx_t_6) {
        __Pyx_GIVEREF(__pyx_t_6); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_6); __pyx_t_6 = NULL;
      }
      __Pyx_INCREF(__pyx_v_pyobj);
      __Pyx_GIVEREF(__pyx_v_pyobj);
      PyTuple_SET_ITEM(__pyx_t_4, 0+__pyx_t_5, __pyx_v_pyobj);
      __Pyx_INCREF(__pyx_v_h);
      __Pyx_GIVEREF(__pyx_v_h);
      PyTuple_SET_ITEM(__pyx_t_4, 1+__pyx_t_5, __pyx_v_h);
      __Pyx_INCREF(__pyx_v_w);
      __Pyx_GIVEREF(__pyx_v_w);
      PyTuple_SET_ITEM(__pyx_t_4, 2+__pyx_t_5, __pyx_v_w);
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_1, __pyx_t_4, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 298, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_objs = __pyx_t_3;
    __pyx_t_3 = 0;

    /* "crowdposetools/_mask.pyx":296
 *     elif type(pyobj) == list and len(pyobj[0]) > 4:
 *         objs = frPoly(pyobj, h, w)
 *     elif type(pyobj) == list and type(pyobj[0]) == dict \             # <<<<<<<<<<<<<<
 *         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
 *         objs = frUncompressedRLE(pyobj, h, w)
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":300
 *         objs = frUncompressedRLE(pyobj, h, w)
 *     # encode rle from single python object
 *     elif type(pyobj) == list and len(pyobj) == 4:             # <<<<<<<<<<<<<<
 *         objs = frBbox([pyobj], h, w)[0]
 *     elif type(pyobj) == list and len(pyobj) > 4:
 */
  __pyx_t_3 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_pyobj)), ((PyObject *)(&PyList_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 300, __pyx_L1_error)
  __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 300, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (__pyx_t_7) {
  } else {
    __pyx_t_2 = __pyx_t_7;
    goto __pyx_L12_bool_binop_done;
  }
  __pyx_t_8 = PyObject_Length(__pyx_v_pyobj); if (unlikely(__pyx_t_8 == ((Py_ssize_t)-1))) __PYX_ERR(0, 300, __pyx_L1_error)
  __pyx_t_7 = ((__pyx_t_8 == 4) != 0);
  __pyx_t_2 = __pyx_t_7;
  __pyx_L12_bool_binop_done:;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":301
 *     # encode rle from single python object
 *     elif type(pyobj) == list and len(pyobj) == 4:
 *         objs = frBbox([pyobj], h, w)[0]             # <<<<<<<<<<<<<<
 *     elif type(pyobj) == list and len(pyobj) > 4:
 *         objs = frPoly([pyobj], h, w)[0]
 */
    __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_frBbox); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 301, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_4 = PyList_New(1); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 301, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __Pyx_INCREF(__pyx_v_pyobj);
    __Pyx_GIVEREF(__pyx_v_pyobj);
    PyList_SET_ITEM(__pyx_t_4, 0, __pyx_v_pyobj);
    __pyx_t_6 = NULL;
    __pyx_t_5 = 0;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_1))) {
      __pyx_t_6 = PyMethod_GET_SELF(__pyx_t_1);
      if (likely(__pyx_t_6)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_1);
        __Pyx_INCREF(__pyx_t_6);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_1, function);
        __pyx_t_5 = 1;
      }
    }
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[4] = {__pyx_t_6, __pyx_t_4, __pyx_v_h, __pyx_v_w};
      __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_1, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 301, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[4] = {__pyx_t_6, __pyx_t_4, __pyx_v_h, __pyx_v_w};
      __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_1, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 301, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_6); __pyx_t_6 = 0;
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    } else
    #endif
    {
      __pyx_t_10 = PyTuple_New(3+__pyx_t_5); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 301, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_10);
      if (__pyx_t_6) {
        __Pyx_GIVEREF(__pyx_t_6); PyTuple_SET_ITEM(__pyx_t_10, 0, __pyx_t_6); __pyx_t_6 = NULL;
      }
      __Pyx_GIVEREF(__pyx_t_4);
      PyTuple_SET_ITEM(__pyx_t_10, 0+__pyx_t_5, __pyx_t_4);
      __Pyx_INCREF(__pyx_v_h);
      __Pyx_GIVEREF(__pyx_v_h);
      PyTuple_SET_ITEM(__pyx_t_10, 1+__pyx_t_5, __pyx_v_h);
      __Pyx_INCREF(__pyx_v_w);
      __Pyx_GIVEREF(__pyx_v_w);
      PyTuple_SET_ITEM(__pyx_t_10, 2+__pyx_t_5, __pyx_v_w);
      __pyx_t_4 = 0;
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_1, __pyx_t_10, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 301, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
    }
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = __Pyx_GetItemInt(__pyx_t_3, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 301, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_v_objs = __pyx_t_1;
    __pyx_t_1 = 0;

    /* "crowdposetools/_mask.pyx":300
 *         objs = frUncompressedRLE(pyobj, h, w)
 *     # encode rle from single python object
 *     elif type(pyobj) == list and len(pyobj) == 4:             # <<<<<<<<<<<<<<
 *         objs = frBbox([pyobj], h, w)[0]
 *     elif type(pyobj) == list and len(pyobj) > 4:
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":302
 *     elif type(pyobj) == list and len(pyobj) == 4:
 *         objs = frBbox([pyobj], h, w)[0]
 *     elif type(pyobj) == list and len(pyobj) > 4:             # <<<<<<<<<<<<<<
 *         objs = frPoly([pyobj], h, w)[0]
 *     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
 */
  __pyx_t_1 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_pyobj)), ((PyObject *)(&PyList_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 302, __pyx_L1_error)
  __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_1); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 302, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  if (__pyx_t_7) {
  } else {
    __pyx_t_2 = __pyx_t_7;
    goto __pyx_L14_bool_binop_done;
  }
  __pyx_t_8 = PyObject_Length(__pyx_v_pyobj); if (unlikely(__pyx_t_8 == ((Py_ssize_t)-1))) __PYX_ERR(0, 302, __pyx_L1_error)
  __pyx_t_7 = ((__pyx_t_8 > 4) != 0);
  __pyx_t_2 = __pyx_t_7;
  __pyx_L14_bool_binop_done:;
  if (__pyx_t_2) {

    /* "crowdposetools/_mask.pyx":303
 *         objs = frBbox([pyobj], h, w)[0]
 *     elif type(pyobj) == list and len(pyobj) > 4:
 *         objs = frPoly([pyobj], h, w)[0]             # <<<<<<<<<<<<<<
 *     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
 *         objs = frUncompressedRLE([pyobj], h, w)[0]
 */
    __pyx_t_3 = __Pyx_GetModuleGlobalName(__pyx_n_s_frPoly); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 303, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_10 = PyList_New(1); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 303, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_10);
    __Pyx_INCREF(__pyx_v_pyobj);
    __Pyx_GIVEREF(__pyx_v_pyobj);
    PyList_SET_ITEM(__pyx_t_10, 0, __pyx_v_pyobj);
    __pyx_t_4 = NULL;
    __pyx_t_5 = 0;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_3))) {
      __pyx_t_4 = PyMethod_GET_SELF(__pyx_t_3);
      if (likely(__pyx_t_4)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_3);
        __Pyx_INCREF(__pyx_t_4);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_3, function);
        __pyx_t_5 = 1;
      }
    }
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_4, __pyx_t_10, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 303, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_3)) {
      PyObject *__pyx_temp[4] = {__pyx_t_4, __pyx_t_10, __pyx_v_h, __pyx_v_w};
      __pyx_t_1 = __Pyx_PyCFunction_FastCall(__pyx_t_3, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 303, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_4); __pyx_t_4 = 0;
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_10); __pyx_t_10 = 0;
    } else
    #endif
    {
      __pyx_t_6 = PyTuple_New(3+__pyx_t_5); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 303, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_6);
      if (__pyx_t_4) {
        __Pyx_GIVEREF(__pyx_t_4); PyTuple_SET_ITEM(__pyx_t_6, 0, __pyx_t_4); __pyx_t_4 = NULL;
      }
      __Pyx_GIVEREF(__pyx_t_10);
      PyTuple_SET_ITEM(__pyx_t_6, 0+__pyx_t_5, __pyx_t_10);
      __Pyx_INCREF(__pyx_v_h);
      __Pyx_GIVEREF(__pyx_v_h);
      PyTuple_SET_ITEM(__pyx_t_6, 1+__pyx_t_5, __pyx_v_h);
      __Pyx_INCREF(__pyx_v_w);
      __Pyx_GIVEREF(__pyx_v_w);
      PyTuple_SET_ITEM(__pyx_t_6, 2+__pyx_t_5, __pyx_v_w);
      __pyx_t_10 = 0;
      __pyx_t_1 = __Pyx_PyObject_Call(__pyx_t_3, __pyx_t_6, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 303, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_1);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    }
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_3 = __Pyx_GetItemInt(__pyx_t_1, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 303, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_v_objs = __pyx_t_3;
    __pyx_t_3 = 0;

    /* "crowdposetools/_mask.pyx":302
 *     elif type(pyobj) == list and len(pyobj) == 4:
 *         objs = frBbox([pyobj], h, w)[0]
 *     elif type(pyobj) == list and len(pyobj) > 4:             # <<<<<<<<<<<<<<
 *         objs = frPoly([pyobj], h, w)[0]
 *     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":304
 *     elif type(pyobj) == list and len(pyobj) > 4:
 *         objs = frPoly([pyobj], h, w)[0]
 *     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:             # <<<<<<<<<<<<<<
 *         objs = frUncompressedRLE([pyobj], h, w)[0]
 *     else:
 */
  __pyx_t_3 = PyObject_RichCompare(((PyObject *)Py_TYPE(__pyx_v_pyobj)), ((PyObject *)(&PyDict_Type)), Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 304, __pyx_L1_error)
  __pyx_t_7 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 304, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  if (__pyx_t_7) {
  } else {
    __pyx_t_2 = __pyx_t_7;
    goto __pyx_L16_bool_binop_done;
  }
  __pyx_t_7 = (__Pyx_PySequence_ContainsTF(__pyx_n_s_counts, __pyx_v_pyobj, Py_EQ)); if (unlikely(__pyx_t_7 < 0)) __PYX_ERR(0, 304, __pyx_L1_error)
  __pyx_t_9 = (__pyx_t_7 != 0);
  if (__pyx_t_9) {
  } else {
    __pyx_t_2 = __pyx_t_9;
    goto __pyx_L16_bool_binop_done;
  }
  __pyx_t_9 = (__Pyx_PySequence_ContainsTF(__pyx_n_s_size, __pyx_v_pyobj, Py_EQ)); if (unlikely(__pyx_t_9 < 0)) __PYX_ERR(0, 304, __pyx_L1_error)
  __pyx_t_7 = (__pyx_t_9 != 0);
  __pyx_t_2 = __pyx_t_7;
  __pyx_L16_bool_binop_done:;
  if (likely(__pyx_t_2)) {

    /* "crowdposetools/_mask.pyx":305
 *         objs = frPoly([pyobj], h, w)[0]
 *     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
 *         objs = frUncompressedRLE([pyobj], h, w)[0]             # <<<<<<<<<<<<<<
 *     else:
 *         raise Exception('input type is not supported.')
 */
    __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_frUncompressedRLE); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 305, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_6 = PyList_New(1); if (unlikely(!__pyx_t_6)) __PYX_ERR(0, 305, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_6);
    __Pyx_INCREF(__pyx_v_pyobj);
    __Pyx_GIVEREF(__pyx_v_pyobj);
    PyList_SET_ITEM(__pyx_t_6, 0, __pyx_v_pyobj);
    __pyx_t_10 = NULL;
    __pyx_t_5 = 0;
    if (CYTHON_UNPACK_METHODS && unlikely(PyMethod_Check(__pyx_t_1))) {
      __pyx_t_10 = PyMethod_GET_SELF(__pyx_t_1);
      if (likely(__pyx_t_10)) {
        PyObject* function = PyMethod_GET_FUNCTION(__pyx_t_1);
        __Pyx_INCREF(__pyx_t_10);
        __Pyx_INCREF(function);
        __Pyx_DECREF_SET(__pyx_t_1, function);
        __pyx_t_5 = 1;
      }
    }
    #if CYTHON_FAST_PYCALL
    if (PyFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[4] = {__pyx_t_10, __pyx_t_6, __pyx_v_h, __pyx_v_w};
      __pyx_t_3 = __Pyx_PyFunction_FastCall(__pyx_t_1, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 305, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_10); __pyx_t_10 = 0;
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    } else
    #endif
    #if CYTHON_FAST_PYCCALL
    if (__Pyx_PyFastCFunction_Check(__pyx_t_1)) {
      PyObject *__pyx_temp[4] = {__pyx_t_10, __pyx_t_6, __pyx_v_h, __pyx_v_w};
      __pyx_t_3 = __Pyx_PyCFunction_FastCall(__pyx_t_1, __pyx_temp+1-__pyx_t_5, 3+__pyx_t_5); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 305, __pyx_L1_error)
      __Pyx_XDECREF(__pyx_t_10); __pyx_t_10 = 0;
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
    } else
    #endif
    {
      __pyx_t_4 = PyTuple_New(3+__pyx_t_5); if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 305, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      if (__pyx_t_10) {
        __Pyx_GIVEREF(__pyx_t_10); PyTuple_SET_ITEM(__pyx_t_4, 0, __pyx_t_10); __pyx_t_10 = NULL;
      }
      __Pyx_GIVEREF(__pyx_t_6);
      PyTuple_SET_ITEM(__pyx_t_4, 0+__pyx_t_5, __pyx_t_6);
      __Pyx_INCREF(__pyx_v_h);
      __Pyx_GIVEREF(__pyx_v_h);
      PyTuple_SET_ITEM(__pyx_t_4, 1+__pyx_t_5, __pyx_v_h);
      __Pyx_INCREF(__pyx_v_w);
      __Pyx_GIVEREF(__pyx_v_w);
      PyTuple_SET_ITEM(__pyx_t_4, 2+__pyx_t_5, __pyx_v_w);
      __pyx_t_6 = 0;
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_t_1, __pyx_t_4, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 305, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    }
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __pyx_t_1 = __Pyx_GetItemInt(__pyx_t_3, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 305, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_v_objs = __pyx_t_1;
    __pyx_t_1 = 0;

    /* "crowdposetools/_mask.pyx":304
 *     elif type(pyobj) == list and len(pyobj) > 4:
 *         objs = frPoly([pyobj], h, w)[0]
 *     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:             # <<<<<<<<<<<<<<
 *         objs = frUncompressedRLE([pyobj], h, w)[0]
 *     else:
 */
    goto __pyx_L3;
  }

  /* "crowdposetools/_mask.pyx":307
 *         objs = frUncompressedRLE([pyobj], h, w)[0]
 *     else:
 *         raise Exception('input type is not supported.')             # <<<<<<<<<<<<<<
 *     return objs
 */
  /*else*/ {
    __pyx_t_1 = __Pyx_PyObject_Call(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])), __pyx_tuple__22, NULL); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 307, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_1);
    __Pyx_Raise(__pyx_t_1, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __PYX_ERR(0, 307, __pyx_L1_error)
  }
  __pyx_L3:;

  /* "crowdposetools/_mask.pyx":308
 *     else:
 *         raise Exception('input type is not supported.')
 *     return objs             # <<<<<<<<<<<<<<
 */
  __Pyx_XDECREF(__pyx_r);
  __Pyx_INCREF(__pyx_v_objs);
  __pyx_r = __pyx_v_objs;
  goto __pyx_L0;

  /* "crowdposetools/_mask.pyx":288
 *     return objs
 * 
 * def frPyObjects(pyobj, h, w):             # <<<<<<<<<<<<<<
 *     # encode rle from a list of python objects
 *     if type(pyobj) == np.ndarray:
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_10);
  __Pyx_AddTraceback("crowdposetools._mask.frPyObjects", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF(__pyx_v_objs);
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":215
 *         # experimental exception made for __getbuffer__ and __releasebuffer__
 *         # -- the details of this may change.
 *         def __getbuffer__(ndarray self, Py_buffer* info, int flags):             # <<<<<<<<<<<<<<
 *             # This implementation of getbuffer is geared towards Cython
 *             # requirements, and does not yet fulfill the PEP.
 */

/* Python wrapper */
static CYTHON_UNUSED int __pyx_pw_5numpy_7ndarray_1__getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags); /*proto*/
static CYTHON_UNUSED int __pyx_pw_5numpy_7ndarray_1__getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__getbuffer__ (wrapper)", 0);
  __pyx_r = __pyx_pf_5numpy_7ndarray___getbuffer__(((PyArrayObject *)__pyx_v_self), ((Py_buffer *)__pyx_v_info), ((int)__pyx_v_flags));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static int __pyx_pf_5numpy_7ndarray___getbuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags) {
  int __pyx_v_i;
  int __pyx_v_ndim;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  int __pyx_v_t;
  char *__pyx_v_f;
  PyArray_Descr *__pyx_v_descr = 0;
  int __pyx_v_offset;
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  int __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  int __pyx_t_5;
  int __pyx_t_6;
  PyObject *__pyx_t_7 = NULL;
  char *__pyx_t_8;
  if (__pyx_v_info == NULL) {
    PyErr_SetString(PyExc_BufferError, "PyObject_GetBuffer: view==NULL argument is obsolete");
    return -1;
  }
  __Pyx_RefNannySetupContext("__getbuffer__", 0);
  __pyx_v_info->obj = Py_None; __Pyx_INCREF(Py_None);
  __Pyx_GIVEREF(__pyx_v_info->obj);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":222
 * 
 *             cdef int i, ndim
 *             cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 */
  __pyx_v_endian_detector = 1;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":223
 *             cdef int i, ndim
 *             cdef int endian_detector = 1
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 * 
 *             ndim = PyArray_NDIM(self)
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":225
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 *             ndim = PyArray_NDIM(self)             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 */
  __pyx_v_ndim = PyArray_NDIM(__pyx_v_self);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":227
 *             ndim = PyArray_NDIM(self)
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  __pyx_t_2 = (((__pyx_v_flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L4_bool_binop_done;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":228
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 */
  __pyx_t_2 = ((!(PyArray_CHKFLAGS(__pyx_v_self, NPY_C_CONTIGUOUS) != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L4_bool_binop_done:;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":227
 *             ndim = PyArray_NDIM(self)
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  if (unlikely(__pyx_t_1)) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":229
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
    __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__23, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 229, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_Raise(__pyx_t_3, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __PYX_ERR(2, 229, __pyx_L1_error)

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":227
 *             ndim = PyArray_NDIM(self)
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":231
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  __pyx_t_2 = (((__pyx_v_flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) != 0);
  if (__pyx_t_2) {
  } else {
    __pyx_t_1 = __pyx_t_2;
    goto __pyx_L7_bool_binop_done;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":232
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 */
  __pyx_t_2 = ((!(PyArray_CHKFLAGS(__pyx_v_self, NPY_F_CONTIGUOUS) != 0)) != 0);
  __pyx_t_1 = __pyx_t_2;
  __pyx_L7_bool_binop_done:;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":231
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  if (unlikely(__pyx_t_1)) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":233
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
    __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__24, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 233, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_Raise(__pyx_t_3, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __PYX_ERR(2, 233, __pyx_L1_error)

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":231
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":235
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 *             info.buf = PyArray_DATA(self)             # <<<<<<<<<<<<<<
 *             info.ndim = ndim
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_v_info->buf = PyArray_DATA(__pyx_v_self);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":236
 * 
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim             # <<<<<<<<<<<<<<
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 # Allocate new buffer for strides and shape info.
 */
  __pyx_v_info->ndim = __pyx_v_ndim;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":237
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 */
  __pyx_t_1 = (((sizeof(npy_intp)) != (sizeof(Py_ssize_t))) != 0);
  if (__pyx_t_1) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":240
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>PyObject_Malloc(sizeof(Py_ssize_t) * 2 * <size_t>ndim)             # <<<<<<<<<<<<<<
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 */
    __pyx_v_info->strides = ((Py_ssize_t *)PyObject_Malloc((((sizeof(Py_ssize_t)) * 2) * ((size_t)__pyx_v_ndim))));

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":241
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>PyObject_Malloc(sizeof(Py_ssize_t) * 2 * <size_t>ndim)
 *                 info.shape = info.strides + ndim             # <<<<<<<<<<<<<<
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 */
    __pyx_v_info->shape = (__pyx_v_info->strides + __pyx_v_ndim);

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":242
 *                 info.strides = <Py_ssize_t*>PyObject_Malloc(sizeof(Py_ssize_t) * 2 * <size_t>ndim)
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):             # <<<<<<<<<<<<<<
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 */
    __pyx_t_4 = __pyx_v_ndim;
    __pyx_t_5 = __pyx_t_4;
    for (__pyx_t_6 = 0; __pyx_t_6 < __pyx_t_5; __pyx_t_6+=1) {
      __pyx_v_i = __pyx_t_6;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":243
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]             # <<<<<<<<<<<<<<
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 */
      (__pyx_v_info->strides[__pyx_v_i]) = (PyArray_STRIDES(__pyx_v_self)[__pyx_v_i]);

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":244
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]             # <<<<<<<<<<<<<<
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 */
      (__pyx_v_info->shape[__pyx_v_i]) = (PyArray_DIMS(__pyx_v_self)[__pyx_v_i]);
    }

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":237
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 */
    goto __pyx_L9;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":246
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)             # <<<<<<<<<<<<<<
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 */
  /*else*/ {
    __pyx_v_info->strides = ((Py_ssize_t *)PyArray_STRIDES(__pyx_v_self));

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":247
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)             # <<<<<<<<<<<<<<
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 */
    __pyx_v_info->shape = ((Py_ssize_t *)PyArray_DIMS(__pyx_v_self));
  }
  __pyx_L9:;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":248
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL             # <<<<<<<<<<<<<<
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 */
  __pyx_v_info->suboffsets = NULL;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":249
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)             # <<<<<<<<<<<<<<
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 * 
 */
  __pyx_v_info->itemsize = PyArray_ITEMSIZE(__pyx_v_self);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":250
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)             # <<<<<<<<<<<<<<
 * 
 *             cdef int t
 */
  __pyx_v_info->readonly = (!(PyArray_ISWRITEABLE(__pyx_v_self) != 0));

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":253
 * 
 *             cdef int t
 *             cdef char* f = NULL             # <<<<<<<<<<<<<<
 *             cdef dtype descr = self.descr
 *             cdef int offset
 */
  __pyx_v_f = NULL;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":254
 *             cdef int t
 *             cdef char* f = NULL
 *             cdef dtype descr = self.descr             # <<<<<<<<<<<<<<
 *             cdef int offset
 * 
 */
  __pyx_t_3 = ((PyObject *)__pyx_v_self->descr);
  __Pyx_INCREF(__pyx_t_3);
  __pyx_v_descr = ((PyArray_Descr *)__pyx_t_3);
  __pyx_t_3 = 0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":257
 *             cdef int offset
 * 
 *             info.obj = self             # <<<<<<<<<<<<<<
 * 
 *             if not PyDataType_HASFIELDS(descr):
 */
  __Pyx_INCREF(((PyObject *)__pyx_v_self));
  __Pyx_GIVEREF(((PyObject *)__pyx_v_self));
  __Pyx_GOTREF(__pyx_v_info->obj);
  __Pyx_DECREF(__pyx_v_info->obj);
  __pyx_v_info->obj = ((PyObject *)__pyx_v_self);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":259
 *             info.obj = self
 * 
 *             if not PyDataType_HASFIELDS(descr):             # <<<<<<<<<<<<<<
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 */
  __pyx_t_1 = ((!(PyDataType_HASFIELDS(__pyx_v_descr) != 0)) != 0);
  if (__pyx_t_1) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":260
 * 
 *             if not PyDataType_HASFIELDS(descr):
 *                 t = descr.type_num             # <<<<<<<<<<<<<<
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 */
    __pyx_t_4 = __pyx_v_descr->type_num;
    __pyx_v_t = __pyx_t_4;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":261
 *             if not PyDataType_HASFIELDS(descr):
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_2 = ((__pyx_v_descr->byteorder == '>') != 0);
    if (!__pyx_t_2) {
      goto __pyx_L15_next_or;
    } else {
    }
    __pyx_t_2 = (__pyx_v_little_endian != 0);
    if (!__pyx_t_2) {
    } else {
      __pyx_t_1 = __pyx_t_2;
      goto __pyx_L14_bool_binop_done;
    }
    __pyx_L15_next_or:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":262
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):             # <<<<<<<<<<<<<<
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 */
    __pyx_t_2 = ((__pyx_v_descr->byteorder == '<') != 0);
    if (__pyx_t_2) {
    } else {
      __pyx_t_1 = __pyx_t_2;
      goto __pyx_L14_bool_binop_done;
    }
    __pyx_t_2 = ((!(__pyx_v_little_endian != 0)) != 0);
    __pyx_t_1 = __pyx_t_2;
    __pyx_L14_bool_binop_done:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":261
 *             if not PyDataType_HASFIELDS(descr):
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    if (unlikely(__pyx_t_1)) {

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":263
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__25, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 263, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(2, 263, __pyx_L1_error)

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":261
 *             if not PyDataType_HASFIELDS(descr):
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    }

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":264
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 */
    switch (__pyx_v_t) {
      case NPY_BYTE:
      __pyx_v_f = ((char *)"b");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":265
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 */
      case NPY_UBYTE:
      __pyx_v_f = ((char *)"B");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":266
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 */
      case NPY_SHORT:
      __pyx_v_f = ((char *)"h");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":267
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 */
      case NPY_USHORT:
      __pyx_v_f = ((char *)"H");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":268
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 */
      case NPY_INT:
      __pyx_v_f = ((char *)"i");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":269
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 */
      case NPY_UINT:
      __pyx_v_f = ((char *)"I");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":270
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 */
      case NPY_LONG:
      __pyx_v_f = ((char *)"l");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":271
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 */
      case NPY_ULONG:
      __pyx_v_f = ((char *)"L");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":272
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 */
      case NPY_LONGLONG:
      __pyx_v_f = ((char *)"q");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":273
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 */
      case NPY_ULONGLONG:
      __pyx_v_f = ((char *)"Q");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":274
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 */
      case NPY_FLOAT:
      __pyx_v_f = ((char *)"f");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":275
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 */
      case NPY_DOUBLE:
      __pyx_v_f = ((char *)"d");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":276
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 */
      case NPY_LONGDOUBLE:
      __pyx_v_f = ((char *)"g");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":277
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 */
      case NPY_CFLOAT:
      __pyx_v_f = ((char *)"Zf");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":278
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"
 */
      case NPY_CDOUBLE:
      __pyx_v_f = ((char *)"Zd");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":279
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 */
      case NPY_CLONGDOUBLE:
      __pyx_v_f = ((char *)"Zg");
      break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":280
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"             # <<<<<<<<<<<<<<
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
      case NPY_OBJECT:
      __pyx_v_f = ((char *)"O");
      break;
      default:

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":282
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *                 info.format = f
 *                 return
 */
      __pyx_t_3 = __Pyx_PyInt_From_int(__pyx_v_t); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 282, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_7 = PyUnicode_Format(__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_t_3); if (unlikely(!__pyx_t_7)) __PYX_ERR(2, 282, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_7);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_3 = __Pyx_PyObject_CallOneArg(__pyx_builtin_ValueError, __pyx_t_7); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 282, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(2, 282, __pyx_L1_error)
      break;
    }

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":283
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f             # <<<<<<<<<<<<<<
 *                 return
 *             else:
 */
    __pyx_v_info->format = __pyx_v_f;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":284
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f
 *                 return             # <<<<<<<<<<<<<<
 *             else:
 *                 info.format = <char*>PyObject_Malloc(_buffer_format_string_len)
 */
    __pyx_r = 0;
    goto __pyx_L0;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":259
 *             info.obj = self
 * 
 *             if not PyDataType_HASFIELDS(descr):             # <<<<<<<<<<<<<<
 *                 t = descr.type_num
 *                 if ((descr.byteorder == c'>' and little_endian) or
 */
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":286
 *                 return
 *             else:
 *                 info.format = <char*>PyObject_Malloc(_buffer_format_string_len)             # <<<<<<<<<<<<<<
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0
 */
  /*else*/ {
    __pyx_v_info->format = ((char *)PyObject_Malloc(0xFF));

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":287
 *             else:
 *                 info.format = <char*>PyObject_Malloc(_buffer_format_string_len)
 *                 info.format[0] = c'^' # Native data types, manual alignment             # <<<<<<<<<<<<<<
 *                 offset = 0
 *                 f = _util_dtypestring(descr, info.format + 1,
 */
    (__pyx_v_info->format[0]) = '^';

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":288
 *                 info.format = <char*>PyObject_Malloc(_buffer_format_string_len)
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0             # <<<<<<<<<<<<<<
 *                 f = _util_dtypestring(descr, info.format + 1,
 *                                       info.format + _buffer_format_string_len,
 */
    __pyx_v_offset = 0;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":289
 *                 info.format[0] = c'^' # Native data types, manual alignment
 *                 offset = 0
 *                 f = _util_dtypestring(descr, info.format + 1,             # <<<<<<<<<<<<<<
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)
 */
    __pyx_t_8 = __pyx_f_5numpy__util_dtypestring(__pyx_v_descr, (__pyx_v_info->format + 1), (__pyx_v_info->format + 0xFF), (&__pyx_v_offset)); if (unlikely(__pyx_t_8 == ((char *)NULL))) __PYX_ERR(2, 289, __pyx_L1_error)
    __pyx_v_f = __pyx_t_8;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":292
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)
 *                 f[0] = c'\0' # Terminate format string             # <<<<<<<<<<<<<<
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 */
    (__pyx_v_f[0]) = '\x00';
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":215
 *         # experimental exception made for __getbuffer__ and __releasebuffer__
 *         # -- the details of this may change.
 *         def __getbuffer__(ndarray self, Py_buffer* info, int flags):             # <<<<<<<<<<<<<<
 *             # This implementation of getbuffer is geared towards Cython
 *             # requirements, and does not yet fulfill the PEP.
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_AddTraceback("numpy.ndarray.__getbuffer__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  if (__pyx_v_info->obj != NULL) {
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj); __pyx_v_info->obj = 0;
  }
  goto __pyx_L2;
  __pyx_L0:;
  if (__pyx_v_info->obj == Py_None) {
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj); __pyx_v_info->obj = 0;
  }
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_descr);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":294
 *                 f[0] = c'\0' # Terminate format string
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):             # <<<<<<<<<<<<<<
 *             if PyArray_HASFIELDS(self):
 *                 PyObject_Free(info.format)
 */

/* Python wrapper */
static CYTHON_UNUSED void __pyx_pw_5numpy_7ndarray_3__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info); /*proto*/
static CYTHON_UNUSED void __pyx_pw_5numpy_7ndarray_3__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__releasebuffer__ (wrapper)", 0);
  __pyx_pf_5numpy_7ndarray_2__releasebuffer__(((PyArrayObject *)__pyx_v_self), ((Py_buffer *)__pyx_v_info));

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

static void __pyx_pf_5numpy_7ndarray_2__releasebuffer__(PyArrayObject *__pyx_v_self, Py_buffer *__pyx_v_info) {
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("__releasebuffer__", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":295
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):             # <<<<<<<<<<<<<<
 *                 PyObject_Free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_t_1 = (PyArray_HASFIELDS(__pyx_v_self) != 0);
  if (__pyx_t_1) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":296
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):
 *                 PyObject_Free(info.format)             # <<<<<<<<<<<<<<
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 PyObject_Free(info.strides)
 */
    PyObject_Free(__pyx_v_info->format);

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":295
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):             # <<<<<<<<<<<<<<
 *                 PyObject_Free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":297
 *             if PyArray_HASFIELDS(self):
 *                 PyObject_Free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 PyObject_Free(info.strides)
 *                 # info.shape was stored after info.strides in the same block
 */
  __pyx_t_1 = (((sizeof(npy_intp)) != (sizeof(Py_ssize_t))) != 0);
  if (__pyx_t_1) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":298
 *                 PyObject_Free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 PyObject_Free(info.strides)             # <<<<<<<<<<<<<<
 *                 # info.shape was stored after info.strides in the same block
 * 
 */
    PyObject_Free(__pyx_v_info->strides);

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":297
 *             if PyArray_HASFIELDS(self):
 *                 PyObject_Free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 PyObject_Free(info.strides)
 *                 # info.shape was stored after info.strides in the same block
 */
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":294
 *                 f[0] = c'\0' # Terminate format string
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):             # <<<<<<<<<<<<<<
 *             if PyArray_HASFIELDS(self):
 *                 PyObject_Free(info.format)
 */

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":775
 * ctypedef npy_cdouble     complex_t
 * 
 * cdef inline object PyArray_MultiIterNew1(a):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew1(PyObject *__pyx_v_a) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew1", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":776
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 *     return PyArray_MultiIterNew(1, <void*>a)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(1, ((void *)__pyx_v_a)); if (unlikely(!__pyx_t_1)) __PYX_ERR(2, 776, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":775
 * ctypedef npy_cdouble     complex_t
 * 
 * cdef inline object PyArray_MultiIterNew1(a):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew1", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":778
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew2(PyObject *__pyx_v_a, PyObject *__pyx_v_b) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew2", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":779
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(2, ((void *)__pyx_v_a), ((void *)__pyx_v_b)); if (unlikely(!__pyx_t_1)) __PYX_ERR(2, 779, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":778
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew2", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":781
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew3(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew3", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":782
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(3, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c)); if (unlikely(!__pyx_t_1)) __PYX_ERR(2, 782, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":781
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew3", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":784
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew4(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew4", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":785
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(4, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d)); if (unlikely(!__pyx_t_1)) __PYX_ERR(2, 785, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":784
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew4", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":787
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew5(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d, PyObject *__pyx_v_e) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew5", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":788
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)             # <<<<<<<<<<<<<<
 * 
 * cdef inline tuple PyDataType_SHAPE(dtype d):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(5, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d), ((void *)__pyx_v_e)); if (unlikely(!__pyx_t_1)) __PYX_ERR(2, 788, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":787
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew5", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":790
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 * cdef inline tuple PyDataType_SHAPE(dtype d):             # <<<<<<<<<<<<<<
 *     if PyDataType_HASSUBARRAY(d):
 *         return <tuple>d.subarray.shape
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyDataType_SHAPE(PyArray_Descr *__pyx_v_d) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("PyDataType_SHAPE", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":791
 * 
 * cdef inline tuple PyDataType_SHAPE(dtype d):
 *     if PyDataType_HASSUBARRAY(d):             # <<<<<<<<<<<<<<
 *         return <tuple>d.subarray.shape
 *     else:
 */
  __pyx_t_1 = (PyDataType_HASSUBARRAY(__pyx_v_d) != 0);
  if (__pyx_t_1) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":792
 * cdef inline tuple PyDataType_SHAPE(dtype d):
 *     if PyDataType_HASSUBARRAY(d):
 *         return <tuple>d.subarray.shape             # <<<<<<<<<<<<<<
 *     else:
 *         return ()
 */
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(((PyObject*)__pyx_v_d->subarray->shape));
    __pyx_r = ((PyObject*)__pyx_v_d->subarray->shape);
    goto __pyx_L0;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":791
 * 
 * cdef inline tuple PyDataType_SHAPE(dtype d):
 *     if PyDataType_HASSUBARRAY(d):             # <<<<<<<<<<<<<<
 *         return <tuple>d.subarray.shape
 *     else:
 */
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":794
 *         return <tuple>d.subarray.shape
 *     else:
 *         return ()             # <<<<<<<<<<<<<<
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:
 */
  /*else*/ {
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(__pyx_empty_tuple);
    __pyx_r = __pyx_empty_tuple;
    goto __pyx_L0;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":790
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 * cdef inline tuple PyDataType_SHAPE(dtype d):             # <<<<<<<<<<<<<<
 *     if PyDataType_HASSUBARRAY(d):
 *         return <tuple>d.subarray.shape
 */

  /* function exit code */
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":796
 *         return ()
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:             # <<<<<<<<<<<<<<
 *     # Recursive utility function used in __getbuffer__ to get format
 *     # string. The new location in the format string is returned.
 */

static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *__pyx_v_descr, char *__pyx_v_f, char *__pyx_v_end, int *__pyx_v_offset) {
  PyArray_Descr *__pyx_v_child = 0;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  PyObject *__pyx_v_fields = 0;
  PyObject *__pyx_v_childname = NULL;
  PyObject *__pyx_v_new_offset = NULL;
  PyObject *__pyx_v_t = NULL;
  char *__pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  Py_ssize_t __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  int __pyx_t_5;
  int __pyx_t_6;
  int __pyx_t_7;
  long __pyx_t_8;
  char *__pyx_t_9;
  __Pyx_RefNannySetupContext("_util_dtypestring", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":801
 * 
 *     cdef dtype child
 *     cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 *     cdef tuple fields
 */
  __pyx_v_endian_detector = 1;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":802
 *     cdef dtype child
 *     cdef int endian_detector = 1
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 *     cdef tuple fields
 * 
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":805
 *     cdef tuple fields
 * 
 *     for childname in descr.names:             # <<<<<<<<<<<<<<
 *         fields = descr.fields[childname]
 *         child, new_offset = fields
 */
  if (unlikely(__pyx_v_descr->names == Py_None)) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
    __PYX_ERR(2, 805, __pyx_L1_error)
  }
  __pyx_t_1 = __pyx_v_descr->names; __Pyx_INCREF(__pyx_t_1); __pyx_t_2 = 0;
  for (;;) {
    if (__pyx_t_2 >= PyTuple_GET_SIZE(__pyx_t_1)) break;
    #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    __pyx_t_3 = PyTuple_GET_ITEM(__pyx_t_1, __pyx_t_2); __Pyx_INCREF(__pyx_t_3); __pyx_t_2++; if (unlikely(0 < 0)) __PYX_ERR(2, 805, __pyx_L1_error)
    #else
    __pyx_t_3 = PySequence_ITEM(__pyx_t_1, __pyx_t_2); __pyx_t_2++; if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 805, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    #endif
    __Pyx_XDECREF_SET(__pyx_v_childname, __pyx_t_3);
    __pyx_t_3 = 0;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":806
 * 
 *     for childname in descr.names:
 *         fields = descr.fields[childname]             # <<<<<<<<<<<<<<
 *         child, new_offset = fields
 * 
 */
    if (unlikely(__pyx_v_descr->fields == Py_None)) {
      PyErr_SetString(PyExc_TypeError, "'NoneType' object is not subscriptable");
      __PYX_ERR(2, 806, __pyx_L1_error)
    }
    __pyx_t_3 = __Pyx_PyDict_GetItem(__pyx_v_descr->fields, __pyx_v_childname); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 806, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    if (!(likely(PyTuple_CheckExact(__pyx_t_3))||((__pyx_t_3) == Py_None)||(PyErr_Format(PyExc_TypeError, "Expected %.16s, got %.200s", "tuple", Py_TYPE(__pyx_t_3)->tp_name), 0))) __PYX_ERR(2, 806, __pyx_L1_error)
    __Pyx_XDECREF_SET(__pyx_v_fields, ((PyObject*)__pyx_t_3));
    __pyx_t_3 = 0;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":807
 *     for childname in descr.names:
 *         fields = descr.fields[childname]
 *         child, new_offset = fields             # <<<<<<<<<<<<<<
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 */
    if (likely(__pyx_v_fields != Py_None)) {
      PyObject* sequence = __pyx_v_fields;
      Py_ssize_t size = __Pyx_PySequence_SIZE(sequence);
      if (unlikely(size != 2)) {
        if (size > 2) __Pyx_RaiseTooManyValuesError(2);
        else if (size >= 0) __Pyx_RaiseNeedMoreValuesError(size);
        __PYX_ERR(2, 807, __pyx_L1_error)
      }
      #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
      __pyx_t_3 = PyTuple_GET_ITEM(sequence, 0); 
      __pyx_t_4 = PyTuple_GET_ITEM(sequence, 1); 
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(__pyx_t_4);
      #else
      __pyx_t_3 = PySequence_ITEM(sequence, 0); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 807, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PySequence_ITEM(sequence, 1); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 807, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      #endif
    } else {
      __Pyx_RaiseNoneNotIterableError(); __PYX_ERR(2, 807, __pyx_L1_error)
    }
    if (!(likely(((__pyx_t_3) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_3, __pyx_ptype_5numpy_dtype))))) __PYX_ERR(2, 807, __pyx_L1_error)
    __Pyx_XDECREF_SET(__pyx_v_child, ((PyArray_Descr *)__pyx_t_3));
    __pyx_t_3 = 0;
    __Pyx_XDECREF_SET(__pyx_v_new_offset, __pyx_t_4);
    __pyx_t_4 = 0;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":809
 *         child, new_offset = fields
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:             # <<<<<<<<<<<<<<
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 */
    __pyx_t_4 = __Pyx_PyInt_From_int((__pyx_v_offset[0])); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 809, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_3 = PyNumber_Subtract(__pyx_v_new_offset, __pyx_t_4); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 809, __pyx_L1_error)
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __pyx_t_5 = __Pyx_PyInt_As_int(__pyx_t_3); if (unlikely((__pyx_t_5 == (int)-1) && PyErr_Occurred())) __PYX_ERR(2, 809, __pyx_L1_error)
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_6 = ((((__pyx_v_end - __pyx_v_f) - ((int)__pyx_t_5)) < 15) != 0);
    if (unlikely(__pyx_t_6)) {

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":810
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_RuntimeError, __pyx_tuple__26, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 810, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(2, 810, __pyx_L1_error)

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":809
 *         child, new_offset = fields
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:             # <<<<<<<<<<<<<<
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 */
    }

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":812
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_7 = ((__pyx_v_child->byteorder == '>') != 0);
    if (!__pyx_t_7) {
      goto __pyx_L8_next_or;
    } else {
    }
    __pyx_t_7 = (__pyx_v_little_endian != 0);
    if (!__pyx_t_7) {
    } else {
      __pyx_t_6 = __pyx_t_7;
      goto __pyx_L7_bool_binop_done;
    }
    __pyx_L8_next_or:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":813
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):             # <<<<<<<<<<<<<<
 *             raise ValueError(u"Non-native byte order not supported")
 *             # One could encode it in the format string and have Cython
 */
    __pyx_t_7 = ((__pyx_v_child->byteorder == '<') != 0);
    if (__pyx_t_7) {
    } else {
      __pyx_t_6 = __pyx_t_7;
      goto __pyx_L7_bool_binop_done;
    }
    __pyx_t_7 = ((!(__pyx_v_little_endian != 0)) != 0);
    __pyx_t_6 = __pyx_t_7;
    __pyx_L7_bool_binop_done:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":812
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    if (unlikely(__pyx_t_6)) {

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":814
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
      __pyx_t_3 = __Pyx_PyObject_Call(__pyx_builtin_ValueError, __pyx_tuple__27, NULL); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 814, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_Raise(__pyx_t_3, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __PYX_ERR(2, 814, __pyx_L1_error)

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":812
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == c'>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    }

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":824
 * 
 *         # Output padding bytes
 *         while offset[0] < new_offset:             # <<<<<<<<<<<<<<
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 */
    while (1) {
      __pyx_t_3 = __Pyx_PyInt_From_int((__pyx_v_offset[0])); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 824, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_t_3, __pyx_v_new_offset, Py_LT); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 824, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 824, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (!__pyx_t_6) break;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":825
 *         # Output padding bytes
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte             # <<<<<<<<<<<<<<
 *             f += 1
 *             offset[0] += 1
 */
      (__pyx_v_f[0]) = 0x78;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":826
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte
 *             f += 1             # <<<<<<<<<<<<<<
 *             offset[0] += 1
 * 
 */
      __pyx_v_f = (__pyx_v_f + 1);

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":827
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 *             offset[0] += 1             # <<<<<<<<<<<<<<
 * 
 *         offset[0] += child.itemsize
 */
      __pyx_t_8 = 0;
      (__pyx_v_offset[__pyx_t_8]) = ((__pyx_v_offset[__pyx_t_8]) + 1);
    }

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":829
 *             offset[0] += 1
 * 
 *         offset[0] += child.itemsize             # <<<<<<<<<<<<<<
 * 
 *         if not PyDataType_HASFIELDS(child):
 */
    __pyx_t_8 = 0;
    (__pyx_v_offset[__pyx_t_8]) = ((__pyx_v_offset[__pyx_t_8]) + __pyx_v_child->elsize);

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":831
 *         offset[0] += child.itemsize
 * 
 *         if not PyDataType_HASFIELDS(child):             # <<<<<<<<<<<<<<
 *             t = child.type_num
 *             if end - f < 5:
 */
    __pyx_t_6 = ((!(PyDataType_HASFIELDS(__pyx_v_child) != 0)) != 0);
    if (__pyx_t_6) {

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":832
 * 
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num             # <<<<<<<<<<<<<<
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")
 */
      __pyx_t_4 = __Pyx_PyInt_From_int(__pyx_v_child->type_num); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 832, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_XDECREF_SET(__pyx_v_t, __pyx_t_4);
      __pyx_t_4 = 0;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":833
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num
 *             if end - f < 5:             # <<<<<<<<<<<<<<
 *                 raise RuntimeError(u"Format string allocated too short.")
 * 
 */
      __pyx_t_6 = (((__pyx_v_end - __pyx_v_f) < 5) != 0);
      if (unlikely(__pyx_t_6)) {

        /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":834
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
        __pyx_t_4 = __Pyx_PyObject_Call(__pyx_builtin_RuntimeError, __pyx_tuple__28, NULL); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 834, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_Raise(__pyx_t_4, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
        __PYX_ERR(2, 834, __pyx_L1_error)

        /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":833
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num
 *             if end - f < 5:             # <<<<<<<<<<<<<<
 *                 raise RuntimeError(u"Format string allocated too short.")
 * 
 */
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":837
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_BYTE); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 837, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 837, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 837, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 98;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":838
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"             # <<<<<<<<<<<<<<
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_UBYTE); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 838, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 838, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 838, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 66;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":839
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"             # <<<<<<<<<<<<<<
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_SHORT); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 839, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 839, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 839, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x68;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":840
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"             # <<<<<<<<<<<<<<
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_USHORT); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 840, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 840, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 840, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 72;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":841
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_INT); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 841, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 841, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 841, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x69;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":842
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_UINT); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 842, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 842, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 842, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 73;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":843
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONG); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 843, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 843, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 843, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x6C;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":844
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_ULONG); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 844, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 844, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 844, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 76;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":845
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONGLONG); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 845, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 845, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 845, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x71;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":846
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_ULONGLONG); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 846, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 846, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 846, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 81;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":847
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"             # <<<<<<<<<<<<<<
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_FLOAT); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 847, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 847, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 847, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x66;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":848
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_DOUBLE); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 848, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 848, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 848, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x64;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":849
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"             # <<<<<<<<<<<<<<
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_LONGDOUBLE); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 849, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 849, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 849, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 0x67;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":850
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf             # <<<<<<<<<<<<<<
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CFLOAT); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 850, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 850, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 850, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x66;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":851
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd             # <<<<<<<<<<<<<<
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CDOUBLE); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 851, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 851, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 851, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x64;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":852
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg             # <<<<<<<<<<<<<<
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 */
      __pyx_t_3 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_CLONGDOUBLE); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 852, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_4 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); __Pyx_XGOTREF(__pyx_t_4); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 852, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_4); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 852, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 0x67;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":853
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"             # <<<<<<<<<<<<<<
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
      __pyx_t_4 = __Pyx_PyInt_From_enum__NPY_TYPES(NPY_OBJECT); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 853, __pyx_L1_error)
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_4, Py_EQ); __Pyx_XGOTREF(__pyx_t_3); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 853, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) __PYX_ERR(2, 853, __pyx_L1_error)
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (likely(__pyx_t_6)) {
        (__pyx_v_f[0]) = 79;
        goto __pyx_L15;
      }

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":855
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *             f += 1
 *         else:
 */
      /*else*/ {
        __pyx_t_3 = PyUnicode_Format(__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_v_t); if (unlikely(!__pyx_t_3)) __PYX_ERR(2, 855, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_3);
        __pyx_t_4 = __Pyx_PyObject_CallOneArg(__pyx_builtin_ValueError, __pyx_t_3); if (unlikely(!__pyx_t_4)) __PYX_ERR(2, 855, __pyx_L1_error)
        __Pyx_GOTREF(__pyx_t_4);
        __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
        __Pyx_Raise(__pyx_t_4, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
        __PYX_ERR(2, 855, __pyx_L1_error)
      }
      __pyx_L15:;

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":856
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *             f += 1             # <<<<<<<<<<<<<<
 *         else:
 *             # Cython ignores struct boundary information ("T{...}"),
 */
      __pyx_v_f = (__pyx_v_f + 1);

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":831
 *         offset[0] += child.itemsize
 * 
 *         if not PyDataType_HASFIELDS(child):             # <<<<<<<<<<<<<<
 *             t = child.type_num
 *             if end - f < 5:
 */
      goto __pyx_L13;
    }

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":860
 *             # Cython ignores struct boundary information ("T{...}"),
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)             # <<<<<<<<<<<<<<
 *     return f
 * 
 */
    /*else*/ {
      __pyx_t_9 = __pyx_f_5numpy__util_dtypestring(__pyx_v_child, __pyx_v_f, __pyx_v_end, __pyx_v_offset); if (unlikely(__pyx_t_9 == ((char *)NULL))) __PYX_ERR(2, 860, __pyx_L1_error)
      __pyx_v_f = __pyx_t_9;
    }
    __pyx_L13:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":805
 *     cdef tuple fields
 * 
 *     for childname in descr.names:             # <<<<<<<<<<<<<<
 *         fields = descr.fields[childname]
 *         child, new_offset = fields
 */
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":861
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)
 *     return f             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_f;
  goto __pyx_L0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":796
 *         return ()
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:             # <<<<<<<<<<<<<<
 *     # Recursive utility function used in __getbuffer__ to get format
 *     # string. The new location in the format string is returned.
 */

  /* function exit code */
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_AddTraceback("numpy._util_dtypestring", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_child);
  __Pyx_XDECREF(__pyx_v_fields);
  __Pyx_XDECREF(__pyx_v_childname);
  __Pyx_XDECREF(__pyx_v_new_offset);
  __Pyx_XDECREF(__pyx_v_t);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":977
 * 
 * 
 * cdef inline void set_array_base(ndarray arr, object base):             # <<<<<<<<<<<<<<
 *      cdef PyObject* baseptr
 *      if base is None:
 */

static CYTHON_INLINE void __pyx_f_5numpy_set_array_base(PyArrayObject *__pyx_v_arr, PyObject *__pyx_v_base) {
  PyObject *__pyx_v_baseptr;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  int __pyx_t_2;
  __Pyx_RefNannySetupContext("set_array_base", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":979
 * cdef inline void set_array_base(ndarray arr, object base):
 *      cdef PyObject* baseptr
 *      if base is None:             # <<<<<<<<<<<<<<
 *          baseptr = NULL
 *      else:
 */
  __pyx_t_1 = (__pyx_v_base == Py_None);
  __pyx_t_2 = (__pyx_t_1 != 0);
  if (__pyx_t_2) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":980
 *      cdef PyObject* baseptr
 *      if base is None:
 *          baseptr = NULL             # <<<<<<<<<<<<<<
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 */
    __pyx_v_baseptr = NULL;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":979
 * cdef inline void set_array_base(ndarray arr, object base):
 *      cdef PyObject* baseptr
 *      if base is None:             # <<<<<<<<<<<<<<
 *          baseptr = NULL
 *      else:
 */
    goto __pyx_L3;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":982
 *          baseptr = NULL
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!             # <<<<<<<<<<<<<<
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 */
  /*else*/ {
    Py_INCREF(__pyx_v_base);

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":983
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base             # <<<<<<<<<<<<<<
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr
 */
    __pyx_v_baseptr = ((PyObject *)__pyx_v_base);
  }
  __pyx_L3:;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":984
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)             # <<<<<<<<<<<<<<
 *      arr.base = baseptr
 * 
 */
  Py_XDECREF(__pyx_v_arr->base);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":985
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr             # <<<<<<<<<<<<<<
 * 
 * cdef inline object get_array_base(ndarray arr):
 */
  __pyx_v_arr->base = __pyx_v_baseptr;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":977
 * 
 * 
 * cdef inline void set_array_base(ndarray arr, object base):             # <<<<<<<<<<<<<<
 *      cdef PyObject* baseptr
 *      if base is None:
 */

  /* function exit code */
  __Pyx_RefNannyFinishContext();
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":987
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_get_array_base(PyArrayObject *__pyx_v_arr) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("get_array_base", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":988
 * 
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:             # <<<<<<<<<<<<<<
 *         return None
 *     else:
 */
  __pyx_t_1 = ((__pyx_v_arr->base == NULL) != 0);
  if (__pyx_t_1) {

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":989
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:
 *         return None             # <<<<<<<<<<<<<<
 *     else:
 *         return <object>arr.base
 */
    __Pyx_XDECREF(__pyx_r);
    __pyx_r = Py_None; __Pyx_INCREF(Py_None);
    goto __pyx_L0;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":988
 * 
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:             # <<<<<<<<<<<<<<
 *         return None
 *     else:
 */
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":991
 *         return None
 *     else:
 *         return <object>arr.base             # <<<<<<<<<<<<<<
 * 
 * 
 */
  /*else*/ {
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(((PyObject *)__pyx_v_arr->base));
    __pyx_r = ((PyObject *)__pyx_v_arr->base);
    goto __pyx_L0;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":987
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

  /* function exit code */
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":996
 * # Versions of the import_* functions which are more suitable for
 * # Cython code.
 * cdef inline int import_array() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_array()
 */

static CYTHON_INLINE int __pyx_f_5numpy_import_array(void) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  __Pyx_RefNannySetupContext("import_array", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":997
 * # Cython code.
 * cdef inline int import_array() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_array()
 *     except Exception:
 */
  {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ExceptionSave(&__pyx_t_1, &__pyx_t_2, &__pyx_t_3);
    __Pyx_XGOTREF(__pyx_t_1);
    __Pyx_XGOTREF(__pyx_t_2);
    __Pyx_XGOTREF(__pyx_t_3);
    /*try:*/ {

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":998
 * cdef inline int import_array() except -1:
 *     try:
 *         _import_array()             # <<<<<<<<<<<<<<
 *     except Exception:
 *         raise ImportError("numpy.core.multiarray failed to import")
 */
      __pyx_t_4 = _import_array(); if (unlikely(__pyx_t_4 == ((int)-1))) __PYX_ERR(2, 998, __pyx_L3_error)

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":997
 * # Cython code.
 * cdef inline int import_array() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_array()
 *     except Exception:
 */
    }
    __Pyx_XDECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
    goto __pyx_L8_try_end;
    __pyx_L3_error:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":999
 *     try:
 *         _import_array()
 *     except Exception:             # <<<<<<<<<<<<<<
 *         raise ImportError("numpy.core.multiarray failed to import")
 * 
 */
    __pyx_t_4 = __Pyx_PyErr_ExceptionMatches(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])));
    if (__pyx_t_4) {
      __Pyx_AddTraceback("numpy.import_array", __pyx_clineno, __pyx_lineno, __pyx_filename);
      if (__Pyx_GetException(&__pyx_t_5, &__pyx_t_6, &__pyx_t_7) < 0) __PYX_ERR(2, 999, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_GOTREF(__pyx_t_7);

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1000
 *         _import_array()
 *     except Exception:
 *         raise ImportError("numpy.core.multiarray failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_umath() except -1:
 */
      __pyx_t_8 = __Pyx_PyObject_Call(__pyx_builtin_ImportError, __pyx_tuple__29, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(2, 1000, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_8);
      __Pyx_Raise(__pyx_t_8, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      __PYX_ERR(2, 1000, __pyx_L5_except_error)
    }
    goto __pyx_L5_except_error;
    __pyx_L5_except_error:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":997
 * # Cython code.
 * cdef inline int import_array() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_array()
 *     except Exception:
 */
    __Pyx_XGIVEREF(__pyx_t_1);
    __Pyx_XGIVEREF(__pyx_t_2);
    __Pyx_XGIVEREF(__pyx_t_3);
    __Pyx_ExceptionReset(__pyx_t_1, __pyx_t_2, __pyx_t_3);
    goto __pyx_L1_error;
    __pyx_L8_try_end:;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":996
 * # Versions of the import_* functions which are more suitable for
 * # Cython code.
 * cdef inline int import_array() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_array()
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_AddTraceback("numpy.import_array", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1002
 *         raise ImportError("numpy.core.multiarray failed to import")
 * 
 * cdef inline int import_umath() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

static CYTHON_INLINE int __pyx_f_5numpy_import_umath(void) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  __Pyx_RefNannySetupContext("import_umath", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1003
 * 
 * cdef inline int import_umath() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
  {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ExceptionSave(&__pyx_t_1, &__pyx_t_2, &__pyx_t_3);
    __Pyx_XGOTREF(__pyx_t_1);
    __Pyx_XGOTREF(__pyx_t_2);
    __Pyx_XGOTREF(__pyx_t_3);
    /*try:*/ {

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1004
 * cdef inline int import_umath() except -1:
 *     try:
 *         _import_umath()             # <<<<<<<<<<<<<<
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")
 */
      __pyx_t_4 = _import_umath(); if (unlikely(__pyx_t_4 == ((int)-1))) __PYX_ERR(2, 1004, __pyx_L3_error)

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1003
 * 
 * cdef inline int import_umath() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    }
    __Pyx_XDECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
    goto __pyx_L8_try_end;
    __pyx_L3_error:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1005
 *     try:
 *         _import_umath()
 *     except Exception:             # <<<<<<<<<<<<<<
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 */
    __pyx_t_4 = __Pyx_PyErr_ExceptionMatches(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])));
    if (__pyx_t_4) {
      __Pyx_AddTraceback("numpy.import_umath", __pyx_clineno, __pyx_lineno, __pyx_filename);
      if (__Pyx_GetException(&__pyx_t_5, &__pyx_t_6, &__pyx_t_7) < 0) __PYX_ERR(2, 1005, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_GOTREF(__pyx_t_7);

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1006
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_ufunc() except -1:
 */
      __pyx_t_8 = __Pyx_PyObject_Call(__pyx_builtin_ImportError, __pyx_tuple__30, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(2, 1006, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_8);
      __Pyx_Raise(__pyx_t_8, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      __PYX_ERR(2, 1006, __pyx_L5_except_error)
    }
    goto __pyx_L5_except_error;
    __pyx_L5_except_error:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1003
 * 
 * cdef inline int import_umath() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    __Pyx_XGIVEREF(__pyx_t_1);
    __Pyx_XGIVEREF(__pyx_t_2);
    __Pyx_XGIVEREF(__pyx_t_3);
    __Pyx_ExceptionReset(__pyx_t_1, __pyx_t_2, __pyx_t_3);
    goto __pyx_L1_error;
    __pyx_L8_try_end:;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1002
 *         raise ImportError("numpy.core.multiarray failed to import")
 * 
 * cdef inline int import_umath() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_AddTraceback("numpy.import_umath", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1008
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 * cdef inline int import_ufunc() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

static CYTHON_INLINE int __pyx_f_5numpy_import_ufunc(void) {
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  int __pyx_t_4;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyObject *__pyx_t_8 = NULL;
  __Pyx_RefNannySetupContext("import_ufunc", 0);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1009
 * 
 * cdef inline int import_ufunc() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
  {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ExceptionSave(&__pyx_t_1, &__pyx_t_2, &__pyx_t_3);
    __Pyx_XGOTREF(__pyx_t_1);
    __Pyx_XGOTREF(__pyx_t_2);
    __Pyx_XGOTREF(__pyx_t_3);
    /*try:*/ {

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1010
 * cdef inline int import_ufunc() except -1:
 *     try:
 *         _import_umath()             # <<<<<<<<<<<<<<
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")
 */
      __pyx_t_4 = _import_umath(); if (unlikely(__pyx_t_4 == ((int)-1))) __PYX_ERR(2, 1010, __pyx_L3_error)

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1009
 * 
 * cdef inline int import_ufunc() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    }
    __Pyx_XDECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_XDECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_XDECREF(__pyx_t_3); __pyx_t_3 = 0;
    goto __pyx_L8_try_end;
    __pyx_L3_error:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1011
 *     try:
 *         _import_umath()
 *     except Exception:             # <<<<<<<<<<<<<<
 *         raise ImportError("numpy.core.umath failed to import")
 */
    __pyx_t_4 = __Pyx_PyErr_ExceptionMatches(((PyObject *)(&((PyTypeObject*)PyExc_Exception)[0])));
    if (__pyx_t_4) {
      __Pyx_AddTraceback("numpy.import_ufunc", __pyx_clineno, __pyx_lineno, __pyx_filename);
      if (__Pyx_GetException(&__pyx_t_5, &__pyx_t_6, &__pyx_t_7) < 0) __PYX_ERR(2, 1011, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_GOTREF(__pyx_t_6);
      __Pyx_GOTREF(__pyx_t_7);

      /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1012
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 */
      __pyx_t_8 = __Pyx_PyObject_Call(__pyx_builtin_ImportError, __pyx_tuple__31, NULL); if (unlikely(!__pyx_t_8)) __PYX_ERR(2, 1012, __pyx_L5_except_error)
      __Pyx_GOTREF(__pyx_t_8);
      __Pyx_Raise(__pyx_t_8, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      __PYX_ERR(2, 1012, __pyx_L5_except_error)
    }
    goto __pyx_L5_except_error;
    __pyx_L5_except_error:;

    /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1009
 * 
 * cdef inline int import_ufunc() except -1:
 *     try:             # <<<<<<<<<<<<<<
 *         _import_umath()
 *     except Exception:
 */
    __Pyx_XGIVEREF(__pyx_t_1);
    __Pyx_XGIVEREF(__pyx_t_2);
    __Pyx_XGIVEREF(__pyx_t_3);
    __Pyx_ExceptionReset(__pyx_t_1, __pyx_t_2, __pyx_t_3);
    goto __pyx_L1_error;
    __pyx_L8_try_end:;
  }

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1008
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 * cdef inline int import_ufunc() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

  /* function exit code */
  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_AddTraceback("numpy.import_ufunc", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyObject *__pyx_tp_new_14crowdposetools_5_mask_RLEs(PyTypeObject *t, PyObject *a, PyObject *k) {
  PyObject *o;
  if (likely((t->tp_flags & Py_TPFLAGS_IS_ABSTRACT) == 0)) {
    o = (*t->tp_alloc)(t, 0);
  } else {
    o = (PyObject *) PyBaseObject_Type.tp_new(t, __pyx_empty_tuple, 0);
  }
  if (unlikely(!o)) return 0;
  if (unlikely(__pyx_pw_14crowdposetools_5_mask_4RLEs_1__cinit__(o, a, k) < 0)) goto bad;
  return o;
  bad:
  Py_DECREF(o); o = 0;
  return NULL;
}

static void __pyx_tp_dealloc_14crowdposetools_5_mask_RLEs(PyObject *o) {
  #if CYTHON_USE_TP_FINALIZE
  if (unlikely(PyType_HasFeature(Py_TYPE(o), Py_TPFLAGS_HAVE_FINALIZE) && Py_TYPE(o)->tp_finalize) && (!PyType_IS_GC(Py_TYPE(o)) || !_PyGC_FINALIZED(o))) {
    if (PyObject_CallFinalizerFromDealloc(o)) return;
  }
  #endif
  {
    PyObject *etype, *eval, *etb;
    PyErr_Fetch(&etype, &eval, &etb);
    ++Py_REFCNT(o);
    __pyx_pw_14crowdposetools_5_mask_4RLEs_3__dealloc__(o);
    --Py_REFCNT(o);
    PyErr_Restore(etype, eval, etb);
  }
  (*Py_TYPE(o)->tp_free)(o);
}

static PyObject *__pyx_tp_getattro_14crowdposetools_5_mask_RLEs(PyObject *o, PyObject *n) {
  PyObject *v = __Pyx_PyObject_GenericGetAttr(o, n);
  if (!v && PyErr_ExceptionMatches(PyExc_AttributeError)) {
    PyErr_Clear();
    v = __pyx_pw_14crowdposetools_5_mask_4RLEs_5__getattr__(o, n);
  }
  return v;
}

static PyMethodDef __pyx_methods_14crowdposetools_5_mask_RLEs[] = {
  {"__getattr__", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_4RLEs_5__getattr__, METH_O|METH_COEXIST, 0},
  {"__reduce_cython__", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_4RLEs_7__reduce_cython__, METH_NOARGS, 0},
  {"__setstate_cython__", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_4RLEs_9__setstate_cython__, METH_O, 0},
  {0, 0, 0, 0}
};

static PyTypeObject __pyx_type_14crowdposetools_5_mask_RLEs = {
  PyVarObject_HEAD_INIT(0, 0)
  "crowdposetools._mask.RLEs", /*tp_name*/
  sizeof(struct __pyx_obj_14crowdposetools_5_mask_RLEs), /*tp_basicsize*/
  0, /*tp_itemsize*/
  __pyx_tp_dealloc_14crowdposetools_5_mask_RLEs, /*tp_dealloc*/
  0, /*tp_print*/
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  #if PY_MAJOR_VERSION < 3
  0, /*tp_compare*/
  #endif
  #if PY_MAJOR_VERSION >= 3
  0, /*tp_as_async*/
  #endif
  0, /*tp_repr*/
  0, /*tp_as_number*/
  0, /*tp_as_sequence*/
  0, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  __pyx_tp_getattro_14crowdposetools_5_mask_RLEs, /*tp_getattro*/
  0, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_VERSION_TAG|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_HAVE_NEWBUFFER|Py_TPFLAGS_BASETYPE, /*tp_flags*/
  0, /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  __pyx_methods_14crowdposetools_5_mask_RLEs, /*tp_methods*/
  0, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  0, /*tp_init*/
  0, /*tp_alloc*/
  __pyx_tp_new_14crowdposetools_5_mask_RLEs, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
  0, /*tp_del*/
  0, /*tp_version_tag*/
  #if PY_VERSION_HEX >= 0x030400a1
  0, /*tp_finalize*/
  #endif
};

static PyObject *__pyx_tp_new_14crowdposetools_5_mask_Masks(PyTypeObject *t, PyObject *a, PyObject *k) {
  PyObject *o;
  if (likely((t->tp_flags & Py_TPFLAGS_IS_ABSTRACT) == 0)) {
    o = (*t->tp_alloc)(t, 0);
  } else {
    o = (PyObject *) PyBaseObject_Type.tp_new(t, __pyx_empty_tuple, 0);
  }
  if (unlikely(!o)) return 0;
  if (unlikely(__pyx_pw_14crowdposetools_5_mask_5Masks_1__cinit__(o, a, k) < 0)) goto bad;
  return o;
  bad:
  Py_DECREF(o); o = 0;
  return NULL;
}

static void __pyx_tp_dealloc_14crowdposetools_5_mask_Masks(PyObject *o) {
  #if CYTHON_USE_TP_FINALIZE
  if (unlikely(PyType_HasFeature(Py_TYPE(o), Py_TPFLAGS_HAVE_FINALIZE) && Py_TYPE(o)->tp_finalize) && (!PyType_IS_GC(Py_TYPE(o)) || !_PyGC_FINALIZED(o))) {
    if (PyObject_CallFinalizerFromDealloc(o)) return;
  }
  #endif
  (*Py_TYPE(o)->tp_free)(o);
}

static PyMethodDef __pyx_methods_14crowdposetools_5_mask_Masks[] = {
  {"__array__", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_5Masks_3__array__, METH_NOARGS, 0},
  {"__reduce_cython__", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_5Masks_5__reduce_cython__, METH_NOARGS, 0},
  {"__setstate_cython__", (PyCFunction)__pyx_pw_14crowdposetools_5_mask_5Masks_7__setstate_cython__, METH_O, 0},
  {0, 0, 0, 0}
};

static PyTypeObject __pyx_type_14crowdposetools_5_mask_Masks = {
  PyVarObject_HEAD_INIT(0, 0)
  "crowdposetools._mask.Masks", /*tp_name*/
  sizeof(struct __pyx_obj_14crowdposetools_5_mask_Masks), /*tp_basicsize*/
  0, /*tp_itemsize*/
  __pyx_tp_dealloc_14crowdposetools_5_mask_Masks, /*tp_dealloc*/
  0, /*tp_print*/
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  #if PY_MAJOR_VERSION < 3
  0, /*tp_compare*/
  #endif
  #if PY_MAJOR_VERSION >= 3
  0, /*tp_as_async*/
  #endif
  0, /*tp_repr*/
  0, /*tp_as_number*/
  0, /*tp_as_sequence*/
  0, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_VERSION_TAG|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_HAVE_NEWBUFFER|Py_TPFLAGS_BASETYPE, /*tp_flags*/
  0, /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  __pyx_methods_14crowdposetools_5_mask_Masks, /*tp_methods*/
  0, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  0, /*tp_init*/
  0, /*tp_alloc*/
  __pyx_tp_new_14crowdposetools_5_mask_Masks, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
  0, /*tp_del*/
  0, /*tp_version_tag*/
  #if PY_VERSION_HEX >= 0x030400a1
  0, /*tp_finalize*/
  #endif
};

static PyMethodDef __pyx_methods[] = {
  {0, 0, 0, 0}
};

#if PY_MAJOR_VERSION >= 3
#if CYTHON_PEP489_MULTI_PHASE_INIT
static PyObject* __pyx_pymod_create(PyObject *spec, PyModuleDef *def); /*proto*/
static int __pyx_pymod_exec__mask(PyObject* module); /*proto*/
static PyModuleDef_Slot __pyx_moduledef_slots[] = {
  {Py_mod_create, (void*)__pyx_pymod_create},
  {Py_mod_exec, (void*)__pyx_pymod_exec__mask},
  {0, NULL}
};
#endif

static struct PyModuleDef __pyx_moduledef = {
    PyModuleDef_HEAD_INIT,
    "_mask",
    0, /* m_doc */
  #if CYTHON_PEP489_MULTI_PHASE_INIT
    0, /* m_size */
  #else
    -1, /* m_size */
  #endif
    __pyx_methods /* m_methods */,
  #if CYTHON_PEP489_MULTI_PHASE_INIT
    __pyx_moduledef_slots, /* m_slots */
  #else
    NULL, /* m_reload */
  #endif
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL /* m_free */
};
#endif

static __Pyx_StringTabEntry __pyx_string_tab[] = {
  {&__pyx_n_s_AttributeError, __pyx_k_AttributeError, sizeof(__pyx_k_AttributeError), 0, 0, 1, 1},
  {&__pyx_n_s_F, __pyx_k_F, sizeof(__pyx_k_F), 0, 0, 1, 1},
  {&__pyx_kp_u_Format_string_allocated_too_shor, __pyx_k_Format_string_allocated_too_shor, sizeof(__pyx_k_Format_string_allocated_too_shor), 0, 1, 0, 0},
  {&__pyx_kp_u_Format_string_allocated_too_shor_2, __pyx_k_Format_string_allocated_too_shor_2, sizeof(__pyx_k_Format_string_allocated_too_shor_2), 0, 1, 0, 0},
  {&__pyx_n_s_ImportError, __pyx_k_ImportError, sizeof(__pyx_k_ImportError), 0, 0, 1, 1},
  {&__pyx_n_s_N, __pyx_k_N, sizeof(__pyx_k_N), 0, 0, 1, 1},
  {&__pyx_kp_u_Non_native_byte_order_not_suppor, __pyx_k_Non_native_byte_order_not_suppor, sizeof(__pyx_k_Non_native_byte_order_not_suppor), 0, 1, 0, 0},
  {&__pyx_n_s_PYTHON_VERSION, __pyx_k_PYTHON_VERSION, sizeof(__pyx_k_PYTHON_VERSION), 0, 0, 1, 1},
  {&__pyx_kp_s_Python_version_must_be_2_or_3, __pyx_k_Python_version_must_be_2_or_3, sizeof(__pyx_k_Python_version_must_be_2_or_3), 0, 0, 1, 0},
  {&__pyx_n_s_R, __pyx_k_R, sizeof(__pyx_k_R), 0, 0, 1, 1},
  {&__pyx_n_s_Rs, __pyx_k_Rs, sizeof(__pyx_k_Rs), 0, 0, 1, 1},
  {&__pyx_n_s_RuntimeError, __pyx_k_RuntimeError, sizeof(__pyx_k_RuntimeError), 0, 0, 1, 1},
  {&__pyx_kp_s_The_dt_and_gt_should_have_the_sa, __pyx_k_The_dt_and_gt_should_have_the_sa, sizeof(__pyx_k_The_dt_and_gt_should_have_the_sa), 0, 0, 1, 0},
  {&__pyx_n_s_TypeError, __pyx_k_TypeError, sizeof(__pyx_k_TypeError), 0, 0, 1, 1},
  {&__pyx_n_s_ValueError, __pyx_k_ValueError, sizeof(__pyx_k_ValueError), 0, 0, 1, 1},
  {&__pyx_n_s_a, __pyx_k_a, sizeof(__pyx_k_a), 0, 0, 1, 1},
  {&__pyx_n_s_a_2, __pyx_k_a_2, sizeof(__pyx_k_a_2), 0, 0, 1, 1},
  {&__pyx_n_s_all, __pyx_k_all, sizeof(__pyx_k_all), 0, 0, 1, 1},
  {&__pyx_n_s_area, __pyx_k_area, sizeof(__pyx_k_area), 0, 0, 1, 1},
  {&__pyx_n_s_array, __pyx_k_array, sizeof(__pyx_k_array), 0, 0, 1, 1},
  {&__pyx_n_s_astype, __pyx_k_astype, sizeof(__pyx_k_astype), 0, 0, 1, 1},
  {&__pyx_n_s_author, __pyx_k_author, sizeof(__pyx_k_author), 0, 0, 1, 1},
  {&__pyx_n_s_bb, __pyx_k_bb, sizeof(__pyx_k_bb), 0, 0, 1, 1},
  {&__pyx_n_s_bbIou, __pyx_k_bbIou, sizeof(__pyx_k_bbIou), 0, 0, 1, 1},
  {&__pyx_n_s_bb_2, __pyx_k_bb_2, sizeof(__pyx_k_bb_2), 0, 0, 1, 1},
  {&__pyx_n_s_c_string, __pyx_k_c_string, sizeof(__pyx_k_c_string), 0, 0, 1, 1},
  {&__pyx_n_s_cline_in_traceback, __pyx_k_cline_in_traceback, sizeof(__pyx_k_cline_in_traceback), 0, 0, 1, 1},
  {&__pyx_n_s_cnts, __pyx_k_cnts, sizeof(__pyx_k_cnts), 0, 0, 1, 1},
  {&__pyx_n_s_counts, __pyx_k_counts, sizeof(__pyx_k_counts), 0, 0, 1, 1},
  {&__pyx_n_s_crowdposetools__mask, __pyx_k_crowdposetools__mask, sizeof(__pyx_k_crowdposetools__mask), 0, 0, 1, 1},
  {&__pyx_kp_s_crowdposetools__mask_pyx, __pyx_k_crowdposetools__mask_pyx, sizeof(__pyx_k_crowdposetools__mask_pyx), 0, 0, 1, 0},
  {&__pyx_n_s_data, __pyx_k_data, sizeof(__pyx_k_data), 0, 0, 1, 1},
  {&__pyx_n_s_decode, __pyx_k_decode, sizeof(__pyx_k_decode), 0, 0, 1, 1},
  {&__pyx_n_s_double, __pyx_k_double, sizeof(__pyx_k_double), 0, 0, 1, 1},
  {&__pyx_n_s_dt, __pyx_k_dt, sizeof(__pyx_k_dt), 0, 0, 1, 1},
  {&__pyx_n_s_dtype, __pyx_k_dtype, sizeof(__pyx_k_dtype), 0, 0, 1, 1},
  {&__pyx_n_s_encode, __pyx_k_encode, sizeof(__pyx_k_encode), 0, 0, 1, 1},
  {&__pyx_n_s_enumerate, __pyx_k_enumerate, sizeof(__pyx_k_enumerate), 0, 0, 1, 1},
  {&__pyx_n_s_frBbox, __pyx_k_frBbox, sizeof(__pyx_k_frBbox), 0, 0, 1, 1},
  {&__pyx_n_s_frPoly, __pyx_k_frPoly, sizeof(__pyx_k_frPoly), 0, 0, 1, 1},
  {&__pyx_n_s_frPyObjects, __pyx_k_frPyObjects, sizeof(__pyx_k_frPyObjects), 0, 0, 1, 1},
  {&__pyx_n_s_frString, __pyx_k_frString, sizeof(__pyx_k_frString), 0, 0, 1, 1},
  {&__pyx_n_s_frUncompressedRLE, __pyx_k_frUncompressedRLE, sizeof(__pyx_k_frUncompressedRLE), 0, 0, 1, 1},
  {&__pyx_n_s_getstate, __pyx_k_getstate, sizeof(__pyx_k_getstate), 0, 0, 1, 1},
  {&__pyx_n_s_gt, __pyx_k_gt, sizeof(__pyx_k_gt), 0, 0, 1, 1},
  {&__pyx_n_s_h, __pyx_k_h, sizeof(__pyx_k_h), 0, 0, 1, 1},
  {&__pyx_n_s_i, __pyx_k_i, sizeof(__pyx_k_i), 0, 0, 1, 1},
  {&__pyx_n_s_import, __pyx_k_import, sizeof(__pyx_k_import), 0, 0, 1, 1},
  {&__pyx_kp_s_input_data_type_not_allowed, __pyx_k_input_data_type_not_allowed, sizeof(__pyx_k_input_data_type_not_allowed), 0, 0, 1, 0},
  {&__pyx_kp_s_input_type_is_not_supported, __pyx_k_input_type_is_not_supported, sizeof(__pyx_k_input_type_is_not_supported), 0, 0, 1, 0},
  {&__pyx_n_s_intersect, __pyx_k_intersect, sizeof(__pyx_k_intersect), 0, 0, 1, 1},
  {&__pyx_n_s_iou, __pyx_k_iou, sizeof(__pyx_k_iou), 0, 0, 1, 1},
  {&__pyx_n_s_iouFun, __pyx_k_iouFun, sizeof(__pyx_k_iouFun), 0, 0, 1, 1},
  {&__pyx_n_s_iou_2, __pyx_k_iou_2, sizeof(__pyx_k_iou_2), 0, 0, 1, 1},
  {&__pyx_n_s_iou_locals__bbIou, __pyx_k_iou_locals__bbIou, sizeof(__pyx_k_iou_locals__bbIou), 0, 0, 1, 1},
  {&__pyx_n_s_iou_locals__len, __pyx_k_iou_locals__len, sizeof(__pyx_k_iou_locals__len), 0, 0, 1, 1},
  {&__pyx_n_s_iou_locals__preproc, __pyx_k_iou_locals__preproc, sizeof(__pyx_k_iou_locals__preproc), 0, 0, 1, 1},
  {&__pyx_n_s_iou_locals__rleIou, __pyx_k_iou_locals__rleIou, sizeof(__pyx_k_iou_locals__rleIou), 0, 0, 1, 1},
  {&__pyx_n_s_isbox, __pyx_k_isbox, sizeof(__pyx_k_isbox), 0, 0, 1, 1},
  {&__pyx_n_s_iscrowd, __pyx_k_iscrowd, sizeof(__pyx_k_iscrowd), 0, 0, 1, 1},
  {&__pyx_n_s_isrle, __pyx_k_isrle, sizeof(__pyx_k_isrle), 0, 0, 1, 1},
  {&__pyx_n_s_j, __pyx_k_j, sizeof(__pyx_k_j), 0, 0, 1, 1},
  {&__pyx_n_s_len, __pyx_k_len, sizeof(__pyx_k_len), 0, 0, 1, 1},
  {&__pyx_kp_s_list_input_can_be_bounding_box_N, __pyx_k_list_input_can_be_bounding_box_N, sizeof(__pyx_k_list_input_can_be_bounding_box_N), 0, 0, 1, 0},
  {&__pyx_n_s_m, __pyx_k_m, sizeof(__pyx_k_m), 0, 0, 1, 1},
  {&__pyx_n_s_main, __pyx_k_main, sizeof(__pyx_k_main), 0, 0, 1, 1},
  {&__pyx_n_s_mask, __pyx_k_mask, sizeof(__pyx_k_mask), 0, 0, 1, 1},
  {&__pyx_n_s_masks, __pyx_k_masks, sizeof(__pyx_k_masks), 0, 0, 1, 1},
  {&__pyx_n_s_merge, __pyx_k_merge, sizeof(__pyx_k_merge), 0, 0, 1, 1},
  {&__pyx_n_s_n, __pyx_k_n, sizeof(__pyx_k_n), 0, 0, 1, 1},
  {&__pyx_n_s_name, __pyx_k_name, sizeof(__pyx_k_name), 0, 0, 1, 1},
  {&__pyx_kp_u_ndarray_is_not_C_contiguous, __pyx_k_ndarray_is_not_C_contiguous, sizeof(__pyx_k_ndarray_is_not_C_contiguous), 0, 1, 0, 0},
  {&__pyx_kp_u_ndarray_is_not_Fortran_contiguou, __pyx_k_ndarray_is_not_Fortran_contiguou, sizeof(__pyx_k_ndarray_is_not_Fortran_contiguou), 0, 1, 0, 0},
  {&__pyx_kp_s_no_default___reduce___due_to_non, __pyx_k_no_default___reduce___due_to_non, sizeof(__pyx_k_no_default___reduce___due_to_non), 0, 0, 1, 0},
  {&__pyx_n_s_np, __pyx_k_np, sizeof(__pyx_k_np), 0, 0, 1, 1},
  {&__pyx_n_s_np_poly, __pyx_k_np_poly, sizeof(__pyx_k_np_poly), 0, 0, 1, 1},
  {&__pyx_n_s_numpy, __pyx_k_numpy, sizeof(__pyx_k_numpy), 0, 0, 1, 1},
  {&__pyx_kp_s_numpy_core_multiarray_failed_to, __pyx_k_numpy_core_multiarray_failed_to, sizeof(__pyx_k_numpy_core_multiarray_failed_to), 0, 0, 1, 0},
  {&__pyx_kp_s_numpy_core_umath_failed_to_impor, __pyx_k_numpy_core_umath_failed_to_impor, sizeof(__pyx_k_numpy_core_umath_failed_to_impor), 0, 0, 1, 0},
  {&__pyx_kp_s_numpy_ndarray_input_is_only_for, __pyx_k_numpy_ndarray_input_is_only_for, sizeof(__pyx_k_numpy_ndarray_input_is_only_for), 0, 0, 1, 0},
  {&__pyx_n_s_obj, __pyx_k_obj, sizeof(__pyx_k_obj), 0, 0, 1, 1},
  {&__pyx_n_s_objs, __pyx_k_objs, sizeof(__pyx_k_objs), 0, 0, 1, 1},
  {&__pyx_n_s_order, __pyx_k_order, sizeof(__pyx_k_order), 0, 0, 1, 1},
  {&__pyx_n_s_p, __pyx_k_p, sizeof(__pyx_k_p), 0, 0, 1, 1},
  {&__pyx_n_s_poly, __pyx_k_poly, sizeof(__pyx_k_poly), 0, 0, 1, 1},
  {&__pyx_n_s_preproc, __pyx_k_preproc, sizeof(__pyx_k_preproc), 0, 0, 1, 1},
  {&__pyx_n_s_py_string, __pyx_k_py_string, sizeof(__pyx_k_py_string), 0, 0, 1, 1},
  {&__pyx_n_s_pyiscrowd, __pyx_k_pyiscrowd, sizeof(__pyx_k_pyiscrowd), 0, 0, 1, 1},
  {&__pyx_n_s_pyobj, __pyx_k_pyobj, sizeof(__pyx_k_pyobj), 0, 0, 1, 1},
  {&__pyx_n_s_range, __pyx_k_range, sizeof(__pyx_k_range), 0, 0, 1, 1},
  {&__pyx_n_s_reduce, __pyx_k_reduce, sizeof(__pyx_k_reduce), 0, 0, 1, 1},
  {&__pyx_n_s_reduce_cython, __pyx_k_reduce_cython, sizeof(__pyx_k_reduce_cython), 0, 0, 1, 1},
  {&__pyx_n_s_reduce_ex, __pyx_k_reduce_ex, sizeof(__pyx_k_reduce_ex), 0, 0, 1, 1},
  {&__pyx_n_s_reshape, __pyx_k_reshape, sizeof(__pyx_k_reshape), 0, 0, 1, 1},
  {&__pyx_n_s_rleIou, __pyx_k_rleIou, sizeof(__pyx_k_rleIou), 0, 0, 1, 1},
  {&__pyx_n_s_rleObjs, __pyx_k_rleObjs, sizeof(__pyx_k_rleObjs), 0, 0, 1, 1},
  {&__pyx_n_s_setstate, __pyx_k_setstate, sizeof(__pyx_k_setstate), 0, 0, 1, 1},
  {&__pyx_n_s_setstate_cython, __pyx_k_setstate_cython, sizeof(__pyx_k_setstate_cython), 0, 0, 1, 1},
  {&__pyx_n_s_shape, __pyx_k_shape, sizeof(__pyx_k_shape), 0, 0, 1, 1},
  {&__pyx_n_s_size, __pyx_k_size, sizeof(__pyx_k_size), 0, 0, 1, 1},
  {&__pyx_n_s_sys, __pyx_k_sys, sizeof(__pyx_k_sys), 0, 0, 1, 1},
  {&__pyx_n_s_test, __pyx_k_test, sizeof(__pyx_k_test), 0, 0, 1, 1},
  {&__pyx_n_s_toBbox, __pyx_k_toBbox, sizeof(__pyx_k_toBbox), 0, 0, 1, 1},
  {&__pyx_n_s_toString, __pyx_k_toString, sizeof(__pyx_k_toString), 0, 0, 1, 1},
  {&__pyx_n_s_tsungyi, __pyx_k_tsungyi, sizeof(__pyx_k_tsungyi), 0, 0, 1, 1},
  {&__pyx_n_s_ucRles, __pyx_k_ucRles, sizeof(__pyx_k_ucRles), 0, 0, 1, 1},
  {&__pyx_n_s_uint32, __pyx_k_uint32, sizeof(__pyx_k_uint32), 0, 0, 1, 1},
  {&__pyx_n_s_uint8, __pyx_k_uint8, sizeof(__pyx_k_uint8), 0, 0, 1, 1},
  {&__pyx_kp_u_unknown_dtype_code_in_numpy_pxd, __pyx_k_unknown_dtype_code_in_numpy_pxd, sizeof(__pyx_k_unknown_dtype_code_in_numpy_pxd), 0, 1, 0, 0},
  {&__pyx_kp_s_unrecognized_type_The_following, __pyx_k_unrecognized_type_The_following, sizeof(__pyx_k_unrecognized_type_The_following), 0, 0, 1, 0},
  {&__pyx_n_s_utf8, __pyx_k_utf8, sizeof(__pyx_k_utf8), 0, 0, 1, 1},
  {&__pyx_n_s_version_info, __pyx_k_version_info, sizeof(__pyx_k_version_info), 0, 0, 1, 1},
  {&__pyx_n_s_w, __pyx_k_w, sizeof(__pyx_k_w), 0, 0, 1, 1},
  {&__pyx_n_s_zeros, __pyx_k_zeros, sizeof(__pyx_k_zeros), 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0}
};
static int __Pyx_InitCachedBuiltins(void) {
  __pyx_builtin_range = __Pyx_GetBuiltinName(__pyx_n_s_range); if (!__pyx_builtin_range) __PYX_ERR(0, 67, __pyx_L1_error)
  __pyx_builtin_AttributeError = __Pyx_GetBuiltinName(__pyx_n_s_AttributeError); if (!__pyx_builtin_AttributeError) __PYX_ERR(0, 73, __pyx_L1_error)
  __pyx_builtin_TypeError = __Pyx_GetBuiltinName(__pyx_n_s_TypeError); if (!__pyx_builtin_TypeError) __PYX_ERR(1, 2, __pyx_L1_error)
  __pyx_builtin_enumerate = __Pyx_GetBuiltinName(__pyx_n_s_enumerate); if (!__pyx_builtin_enumerate) __PYX_ERR(0, 124, __pyx_L1_error)
  __pyx_builtin_ValueError = __Pyx_GetBuiltinName(__pyx_n_s_ValueError); if (!__pyx_builtin_ValueError) __PYX_ERR(2, 229, __pyx_L1_error)
  __pyx_builtin_RuntimeError = __Pyx_GetBuiltinName(__pyx_n_s_RuntimeError); if (!__pyx_builtin_RuntimeError) __PYX_ERR(2, 810, __pyx_L1_error)
  __pyx_builtin_ImportError = __Pyx_GetBuiltinName(__pyx_n_s_ImportError); if (!__pyx_builtin_ImportError) __PYX_ERR(2, 1000, __pyx_L1_error)
  return 0;
  __pyx_L1_error:;
  return -1;
}

static int __Pyx_InitCachedConstants(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_InitCachedConstants", 0);

  /* "(tree fragment)":2
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */
  __pyx_tuple_ = PyTuple_Pack(1, __pyx_kp_s_no_default___reduce___due_to_non); if (unlikely(!__pyx_tuple_)) __PYX_ERR(1, 2, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple_);
  __Pyx_GIVEREF(__pyx_tuple_);

  /* "(tree fragment)":4
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 */
  __pyx_tuple__2 = PyTuple_Pack(1, __pyx_kp_s_no_default___reduce___due_to_non); if (unlikely(!__pyx_tuple__2)) __PYX_ERR(1, 4, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__2);
  __Pyx_GIVEREF(__pyx_tuple__2);

  /* "(tree fragment)":2
 * def __reduce_cython__(self):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 */
  __pyx_tuple__3 = PyTuple_Pack(1, __pyx_kp_s_no_default___reduce___due_to_non); if (unlikely(!__pyx_tuple__3)) __PYX_ERR(1, 2, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__3);
  __Pyx_GIVEREF(__pyx_tuple__3);

  /* "(tree fragment)":4
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")
 * def __setstate_cython__(self, __pyx_state):
 *     raise TypeError("no default __reduce__ due to non-trivial __cinit__")             # <<<<<<<<<<<<<<
 */
  __pyx_tuple__4 = PyTuple_Pack(1, __pyx_kp_s_no_default___reduce___due_to_non); if (unlikely(!__pyx_tuple__4)) __PYX_ERR(1, 4, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__4);
  __Pyx_GIVEREF(__pyx_tuple__4);

  /* "crowdposetools/_mask.pyx":126
 *     for i, obj in enumerate(rleObjs):
 *         if PYTHON_VERSION == 2:
 *             py_string = str(obj['counts']).encode('utf8')             # <<<<<<<<<<<<<<
 *         elif PYTHON_VERSION == 3:
 *             py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
 */
  __pyx_tuple__5 = PyTuple_Pack(1, __pyx_n_s_utf8); if (unlikely(!__pyx_tuple__5)) __PYX_ERR(0, 126, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__5);
  __Pyx_GIVEREF(__pyx_tuple__5);

  /* "crowdposetools/_mask.pyx":130
 *             py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
 *         else:
 *             raise Exception('Python version must be 2 or 3')             # <<<<<<<<<<<<<<
 *         c_string = py_string
 *         rleFrString( <RLE*> &Rs._R[i], <char*> c_string, obj['size'][0], obj['size'][1] )
 */
  __pyx_tuple__6 = PyTuple_Pack(1, __pyx_kp_s_Python_version_must_be_2_or_3); if (unlikely(!__pyx_tuple__6)) __PYX_ERR(0, 130, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__6);
  __Pyx_GIVEREF(__pyx_tuple__6);

  /* "crowdposetools/_mask.pyx":154
 * def merge(rleObjs, intersect=0):
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef RLEs R = RLEs(1)             # <<<<<<<<<<<<<<
 *     rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)
 *     obj = _toString(R)[0]
 */
  __pyx_tuple__7 = PyTuple_Pack(1, __pyx_int_1); if (unlikely(!__pyx_tuple__7)) __PYX_ERR(0, 154, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__7);
  __Pyx_GIVEREF(__pyx_tuple__7);

  /* "crowdposetools/_mask.pyx":180
 *             # check if it's Nx4 bbox
 *             if not len(objs.shape) == 2 or not objs.shape[1] == 4:
 *                 raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')             # <<<<<<<<<<<<<<
 *             objs = objs.astype(np.double)
 *         elif type(objs) == list:
 */
  __pyx_tuple__8 = PyTuple_Pack(1, __pyx_kp_s_numpy_ndarray_input_is_only_for); if (unlikely(!__pyx_tuple__8)) __PYX_ERR(0, 180, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__8);
  __Pyx_GIVEREF(__pyx_tuple__8);

  /* "crowdposetools/_mask.pyx":193
 *                 objs = _frString(objs)
 *             else:
 *                 raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')             # <<<<<<<<<<<<<<
 *         else:
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
 */
  __pyx_tuple__9 = PyTuple_Pack(1, __pyx_kp_s_list_input_can_be_bounding_box_N); if (unlikely(!__pyx_tuple__9)) __PYX_ERR(0, 193, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__9);
  __Pyx_GIVEREF(__pyx_tuple__9);

  /* "crowdposetools/_mask.pyx":195
 *                 raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
 *         else:
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')             # <<<<<<<<<<<<<<
 *         return objs
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 */
  __pyx_tuple__10 = PyTuple_Pack(1, __pyx_kp_s_unrecognized_type_The_following); if (unlikely(!__pyx_tuple__10)) __PYX_ERR(0, 195, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__10);
  __Pyx_GIVEREF(__pyx_tuple__10);

  /* "crowdposetools/_mask.pyx":172
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):
 *     def _preproc(objs):             # <<<<<<<<<<<<<<
 *         if len(objs) == 0:
 *             return objs
 */
  __pyx_tuple__11 = PyTuple_Pack(4, __pyx_n_s_objs, __pyx_n_s_isbox, __pyx_n_s_isrle, __pyx_n_s_obj); if (unlikely(!__pyx_tuple__11)) __PYX_ERR(0, 172, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__11);
  __Pyx_GIVEREF(__pyx_tuple__11);
  __pyx_codeobj__12 = (PyObject*)__Pyx_PyCode_New(1, 0, 4, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__11, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_preproc, 172, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__12)) __PYX_ERR(0, 172, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":197
 *             raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
 *         return objs
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 */
  __pyx_tuple__13 = PyTuple_Pack(6, __pyx_n_s_dt, __pyx_n_s_gt, __pyx_n_s_iscrowd, __pyx_n_s_m, __pyx_n_s_n, __pyx_n_s_iou); if (unlikely(!__pyx_tuple__13)) __PYX_ERR(0, 197, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__13);
  __Pyx_GIVEREF(__pyx_tuple__13);
  __pyx_codeobj__14 = (PyObject*)__Pyx_PyCode_New(6, 0, 6, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__13, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_rleIou, 197, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__14)) __PYX_ERR(0, 197, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":199
 *     def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
 *         rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):             # <<<<<<<<<<<<<<
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):
 */
  __pyx_tuple__15 = PyTuple_Pack(6, __pyx_n_s_dt, __pyx_n_s_gt, __pyx_n_s_iscrowd, __pyx_n_s_m, __pyx_n_s_n, __pyx_n_s_iou); if (unlikely(!__pyx_tuple__15)) __PYX_ERR(0, 199, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__15);
  __Pyx_GIVEREF(__pyx_tuple__15);
  __pyx_codeobj__16 = (PyObject*)__Pyx_PyCode_New(6, 0, 6, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__15, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_bbIou, 199, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__16)) __PYX_ERR(0, 199, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":201
 *     def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
 *         bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
 *     def _len(obj):             # <<<<<<<<<<<<<<
 *         cdef siz N = 0
 *         if type(obj) == RLEs:
 */
  __pyx_tuple__17 = PyTuple_Pack(2, __pyx_n_s_obj, __pyx_n_s_N); if (unlikely(!__pyx_tuple__17)) __PYX_ERR(0, 201, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__17);
  __Pyx_GIVEREF(__pyx_tuple__17);
  __pyx_codeobj__18 = (PyObject*)__Pyx_PyCode_New(1, 0, 2, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__17, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_len, 201, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__18)) __PYX_ERR(0, 201, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":221
 *         return []
 *     if not type(dt) == type(gt):
 *         raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')             # <<<<<<<<<<<<<<
 * 
 *     # define local variables
 */
  __pyx_tuple__19 = PyTuple_Pack(1, __pyx_kp_s_The_dt_and_gt_should_have_the_sa); if (unlikely(!__pyx_tuple__19)) __PYX_ERR(0, 221, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__19);
  __Pyx_GIVEREF(__pyx_tuple__19);

  /* "crowdposetools/_mask.pyx":232
 *         _iouFun = _bbIou
 *     else:
 *         raise Exception('input data type not allowed.')             # <<<<<<<<<<<<<<
 *     _iou = <double*> malloc(m*n* sizeof(double))
 *     iou = np.zeros((m*n, ), dtype=np.double)
 */
  __pyx_tuple__20 = PyTuple_Pack(1, __pyx_kp_s_input_data_type_not_allowed); if (unlikely(!__pyx_tuple__20)) __PYX_ERR(0, 232, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__20);
  __Pyx_GIVEREF(__pyx_tuple__20);

  /* "crowdposetools/_mask.pyx":277
 *     objs = []
 *     for i in range(n):
 *         Rs = RLEs(1)             # <<<<<<<<<<<<<<
 *         cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)
 *         # time for malloc can be saved here but it's fine
 */
  __pyx_tuple__21 = PyTuple_Pack(1, __pyx_int_1); if (unlikely(!__pyx_tuple__21)) __PYX_ERR(0, 277, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__21);
  __Pyx_GIVEREF(__pyx_tuple__21);

  /* "crowdposetools/_mask.pyx":307
 *         objs = frUncompressedRLE([pyobj], h, w)[0]
 *     else:
 *         raise Exception('input type is not supported.')             # <<<<<<<<<<<<<<
 *     return objs
 */
  __pyx_tuple__22 = PyTuple_Pack(1, __pyx_kp_s_input_type_is_not_supported); if (unlikely(!__pyx_tuple__22)) __PYX_ERR(0, 307, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__22);
  __Pyx_GIVEREF(__pyx_tuple__22);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":229
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
  __pyx_tuple__23 = PyTuple_Pack(1, __pyx_kp_u_ndarray_is_not_C_contiguous); if (unlikely(!__pyx_tuple__23)) __PYX_ERR(2, 229, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__23);
  __Pyx_GIVEREF(__pyx_tuple__23);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":233
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
  __pyx_tuple__24 = PyTuple_Pack(1, __pyx_kp_u_ndarray_is_not_Fortran_contiguou); if (unlikely(!__pyx_tuple__24)) __PYX_ERR(2, 233, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__24);
  __Pyx_GIVEREF(__pyx_tuple__24);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":263
 *                 if ((descr.byteorder == c'>' and little_endian) or
 *                     (descr.byteorder == c'<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
  __pyx_tuple__25 = PyTuple_Pack(1, __pyx_kp_u_Non_native_byte_order_not_suppor); if (unlikely(!__pyx_tuple__25)) __PYX_ERR(2, 263, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__25);
  __Pyx_GIVEREF(__pyx_tuple__25);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":810
 * 
 *         if (end - f) - <int>(new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == c'>' and little_endian) or
 */
  __pyx_tuple__26 = PyTuple_Pack(1, __pyx_kp_u_Format_string_allocated_too_shor); if (unlikely(!__pyx_tuple__26)) __PYX_ERR(2, 810, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__26);
  __Pyx_GIVEREF(__pyx_tuple__26);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":814
 *         if ((child.byteorder == c'>' and little_endian) or
 *             (child.byteorder == c'<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
  __pyx_tuple__27 = PyTuple_Pack(1, __pyx_kp_u_Non_native_byte_order_not_suppor); if (unlikely(!__pyx_tuple__27)) __PYX_ERR(2, 814, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__27);
  __Pyx_GIVEREF(__pyx_tuple__27);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":834
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
  __pyx_tuple__28 = PyTuple_Pack(1, __pyx_kp_u_Format_string_allocated_too_shor_2); if (unlikely(!__pyx_tuple__28)) __PYX_ERR(2, 834, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__28);
  __Pyx_GIVEREF(__pyx_tuple__28);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1000
 *         _import_array()
 *     except Exception:
 *         raise ImportError("numpy.core.multiarray failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_umath() except -1:
 */
  __pyx_tuple__29 = PyTuple_Pack(1, __pyx_kp_s_numpy_core_multiarray_failed_to); if (unlikely(!__pyx_tuple__29)) __PYX_ERR(2, 1000, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__29);
  __Pyx_GIVEREF(__pyx_tuple__29);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1006
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 * 
 * cdef inline int import_ufunc() except -1:
 */
  __pyx_tuple__30 = PyTuple_Pack(1, __pyx_kp_s_numpy_core_umath_failed_to_impor); if (unlikely(!__pyx_tuple__30)) __PYX_ERR(2, 1006, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__30);
  __Pyx_GIVEREF(__pyx_tuple__30);

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1012
 *         _import_umath()
 *     except Exception:
 *         raise ImportError("numpy.core.umath failed to import")             # <<<<<<<<<<<<<<
 */
  __pyx_tuple__31 = PyTuple_Pack(1, __pyx_kp_s_numpy_core_umath_failed_to_impor); if (unlikely(!__pyx_tuple__31)) __PYX_ERR(2, 1012, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__31);
  __Pyx_GIVEREF(__pyx_tuple__31);

  /* "crowdposetools/_mask.pyx":103
 * 
 * # internal conversion from Python RLEs object to compressed RLE format
 * def _toString(RLEs Rs):             # <<<<<<<<<<<<<<
 *     cdef siz n = Rs.n
 *     cdef bytes py_string
 */
  __pyx_tuple__32 = PyTuple_Pack(6, __pyx_n_s_Rs, __pyx_n_s_n, __pyx_n_s_py_string, __pyx_n_s_c_string, __pyx_n_s_objs, __pyx_n_s_i); if (unlikely(!__pyx_tuple__32)) __PYX_ERR(0, 103, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__32);
  __Pyx_GIVEREF(__pyx_tuple__32);
  __pyx_codeobj__33 = (PyObject*)__Pyx_PyCode_New(1, 0, 6, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__32, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_toString, 103, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__33)) __PYX_ERR(0, 103, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":119
 * 
 * # internal conversion from compressed RLE format to Python RLEs object
 * def _frString(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef siz n = len(rleObjs)
 *     Rs = RLEs(n)
 */
  __pyx_tuple__34 = PyTuple_Pack(7, __pyx_n_s_rleObjs, __pyx_n_s_n, __pyx_n_s_Rs, __pyx_n_s_py_string, __pyx_n_s_c_string, __pyx_n_s_i, __pyx_n_s_obj); if (unlikely(!__pyx_tuple__34)) __PYX_ERR(0, 119, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__34);
  __Pyx_GIVEREF(__pyx_tuple__34);
  __pyx_codeobj__35 = (PyObject*)__Pyx_PyCode_New(1, 0, 7, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__34, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_frString, 119, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__35)) __PYX_ERR(0, 119, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":137
 * # encode mask to RLEs objects
 * # list of RLE string can be generated by RLEs member function
 * def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):             # <<<<<<<<<<<<<<
 *     h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
 *     cdef RLEs Rs = RLEs(n)
 */
  __pyx_tuple__36 = PyTuple_Pack(6, __pyx_n_s_mask, __pyx_n_s_h, __pyx_n_s_w, __pyx_n_s_n, __pyx_n_s_Rs, __pyx_n_s_objs); if (unlikely(!__pyx_tuple__36)) __PYX_ERR(0, 137, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__36);
  __Pyx_GIVEREF(__pyx_tuple__36);
  __pyx_codeobj__37 = (PyObject*)__Pyx_PyCode_New(1, 0, 6, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__36, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_encode, 137, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__37)) __PYX_ERR(0, 137, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":145
 * 
 * # decode mask from compressed list of RLE string or RLEs object
 * def decode(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
 */
  __pyx_tuple__38 = PyTuple_Pack(6, __pyx_n_s_rleObjs, __pyx_n_s_Rs, __pyx_n_s_h, __pyx_n_s_w, __pyx_n_s_n, __pyx_n_s_masks); if (unlikely(!__pyx_tuple__38)) __PYX_ERR(0, 145, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__38);
  __Pyx_GIVEREF(__pyx_tuple__38);
  __pyx_codeobj__39 = (PyObject*)__Pyx_PyCode_New(1, 0, 6, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__38, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_decode, 145, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__39)) __PYX_ERR(0, 145, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":152
 *     return np.array(masks)
 * 
 * def merge(rleObjs, intersect=0):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef RLEs R = RLEs(1)
 */
  __pyx_tuple__40 = PyTuple_Pack(5, __pyx_n_s_rleObjs, __pyx_n_s_intersect, __pyx_n_s_Rs, __pyx_n_s_R, __pyx_n_s_obj); if (unlikely(!__pyx_tuple__40)) __PYX_ERR(0, 152, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__40);
  __Pyx_GIVEREF(__pyx_tuple__40);
  __pyx_codeobj__41 = (PyObject*)__Pyx_PyCode_New(2, 0, 5, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__40, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_merge, 152, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__41)) __PYX_ERR(0, 152, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":159
 *     return obj
 * 
 * def area(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))
 */
  __pyx_tuple__42 = PyTuple_Pack(5, __pyx_n_s_rleObjs, __pyx_n_s_Rs, __pyx_n_s_a, __pyx_n_s_shape, __pyx_n_s_a_2); if (unlikely(!__pyx_tuple__42)) __PYX_ERR(0, 159, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__42);
  __Pyx_GIVEREF(__pyx_tuple__42);
  __pyx_codeobj__43 = (PyObject*)__Pyx_PyCode_New(1, 0, 5, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__42, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_area, 159, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__43)) __PYX_ERR(0, 159, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":171
 * 
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):             # <<<<<<<<<<<<<<
 *     def _preproc(objs):
 *         if len(objs) == 0:
 */
  __pyx_tuple__44 = PyTuple_Pack(18, __pyx_n_s_dt, __pyx_n_s_gt, __pyx_n_s_pyiscrowd, __pyx_n_s_preproc, __pyx_n_s_preproc, __pyx_n_s_rleIou, __pyx_n_s_rleIou, __pyx_n_s_bbIou, __pyx_n_s_bbIou, __pyx_n_s_len, __pyx_n_s_len, __pyx_n_s_iscrowd, __pyx_n_s_m, __pyx_n_s_n, __pyx_n_s_iou, __pyx_n_s_shape, __pyx_n_s_iouFun, __pyx_n_s_iou_2); if (unlikely(!__pyx_tuple__44)) __PYX_ERR(0, 171, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__44);
  __Pyx_GIVEREF(__pyx_tuple__44);
  __pyx_codeobj__45 = (PyObject*)__Pyx_PyCode_New(3, 0, 18, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__44, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_iou_2, 171, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__45)) __PYX_ERR(0, 171, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":241
 *     return iou.reshape((m,n), order='F')
 * 
 * def toBbox( rleObjs ):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef siz n = Rs.n
 */
  __pyx_tuple__46 = PyTuple_Pack(6, __pyx_n_s_rleObjs, __pyx_n_s_Rs, __pyx_n_s_n, __pyx_n_s_bb_2, __pyx_n_s_shape, __pyx_n_s_bb); if (unlikely(!__pyx_tuple__46)) __PYX_ERR(0, 241, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__46);
  __Pyx_GIVEREF(__pyx_tuple__46);
  __pyx_codeobj__47 = (PyObject*)__Pyx_PyCode_New(1, 0, 6, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__46, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_toBbox, 241, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__47)) __PYX_ERR(0, 241, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":253
 *     return bb
 * 
 * def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef siz n = bb.shape[0]
 *     Rs = RLEs(n)
 */
  __pyx_tuple__48 = PyTuple_Pack(6, __pyx_n_s_bb, __pyx_n_s_h, __pyx_n_s_w, __pyx_n_s_n, __pyx_n_s_Rs, __pyx_n_s_objs); if (unlikely(!__pyx_tuple__48)) __PYX_ERR(0, 253, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__48);
  __Pyx_GIVEREF(__pyx_tuple__48);
  __pyx_codeobj__49 = (PyObject*)__Pyx_PyCode_New(3, 0, 6, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__48, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_frBbox, 253, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__49)) __PYX_ERR(0, 253, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":260
 *     return objs
 * 
 * def frPoly( poly, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.double_t, ndim=1] np_poly
 *     n = len(poly)
 */
  __pyx_tuple__50 = PyTuple_Pack(9, __pyx_n_s_poly, __pyx_n_s_h, __pyx_n_s_w, __pyx_n_s_np_poly, __pyx_n_s_n, __pyx_n_s_Rs, __pyx_n_s_i, __pyx_n_s_p, __pyx_n_s_objs); if (unlikely(!__pyx_tuple__50)) __PYX_ERR(0, 260, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__50);
  __Pyx_GIVEREF(__pyx_tuple__50);
  __pyx_codeobj__51 = (PyObject*)__Pyx_PyCode_New(3, 0, 9, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__50, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_frPoly, 260, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__51)) __PYX_ERR(0, 260, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":270
 *     return objs
 * 
 * def frUncompressedRLE(ucRles, siz h, siz w):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.uint32_t, ndim=1] cnts
 *     cdef RLE R
 */
  __pyx_tuple__52 = PyTuple_Pack(11, __pyx_n_s_ucRles, __pyx_n_s_h, __pyx_n_s_w, __pyx_n_s_cnts, __pyx_n_s_R, __pyx_n_s_data, __pyx_n_s_n, __pyx_n_s_objs, __pyx_n_s_i, __pyx_n_s_Rs, __pyx_n_s_j); if (unlikely(!__pyx_tuple__52)) __PYX_ERR(0, 270, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__52);
  __Pyx_GIVEREF(__pyx_tuple__52);
  __pyx_codeobj__53 = (PyObject*)__Pyx_PyCode_New(3, 0, 11, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__52, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_frUncompressedRLE, 270, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__53)) __PYX_ERR(0, 270, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":288
 *     return objs
 * 
 * def frPyObjects(pyobj, h, w):             # <<<<<<<<<<<<<<
 *     # encode rle from a list of python objects
 *     if type(pyobj) == np.ndarray:
 */
  __pyx_tuple__54 = PyTuple_Pack(4, __pyx_n_s_pyobj, __pyx_n_s_h, __pyx_n_s_w, __pyx_n_s_objs); if (unlikely(!__pyx_tuple__54)) __PYX_ERR(0, 288, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_tuple__54);
  __Pyx_GIVEREF(__pyx_tuple__54);
  __pyx_codeobj__55 = (PyObject*)__Pyx_PyCode_New(3, 0, 4, 0, CO_OPTIMIZED|CO_NEWLOCALS, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__54, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_crowdposetools__mask_pyx, __pyx_n_s_frPyObjects, 288, __pyx_empty_bytes); if (unlikely(!__pyx_codeobj__55)) __PYX_ERR(0, 288, __pyx_L1_error)
  __Pyx_RefNannyFinishContext();
  return 0;
  __pyx_L1_error:;
  __Pyx_RefNannyFinishContext();
  return -1;
}

static int __Pyx_InitGlobals(void) {
  if (__Pyx_InitStrings(__pyx_string_tab) < 0) __PYX_ERR(0, 1, __pyx_L1_error);
  __pyx_int_0 = PyInt_FromLong(0); if (unlikely(!__pyx_int_0)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_1 = PyInt_FromLong(1); if (unlikely(!__pyx_int_1)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_2 = PyInt_FromLong(2); if (unlikely(!__pyx_int_2)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_3 = PyInt_FromLong(3); if (unlikely(!__pyx_int_3)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_int_4 = PyInt_FromLong(4); if (unlikely(!__pyx_int_4)) __PYX_ERR(0, 1, __pyx_L1_error)
  return 0;
  __pyx_L1_error:;
  return -1;
}

static int __Pyx_modinit_global_init_code(void); /*proto*/
static int __Pyx_modinit_variable_export_code(void); /*proto*/
static int __Pyx_modinit_function_export_code(void); /*proto*/
static int __Pyx_modinit_type_init_code(void); /*proto*/
static int __Pyx_modinit_type_import_code(void); /*proto*/
static int __Pyx_modinit_variable_import_code(void); /*proto*/
static int __Pyx_modinit_function_import_code(void); /*proto*/

static int __Pyx_modinit_global_init_code(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_modinit_global_init_code", 0);
  /*--- Global init code ---*/
  __Pyx_RefNannyFinishContext();
  return 0;
}

static int __Pyx_modinit_variable_export_code(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_modinit_variable_export_code", 0);
  /*--- Variable export code ---*/
  __Pyx_RefNannyFinishContext();
  return 0;
}

static int __Pyx_modinit_function_export_code(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_modinit_function_export_code", 0);
  /*--- Function export code ---*/
  __Pyx_RefNannyFinishContext();
  return 0;
}

static int __Pyx_modinit_type_init_code(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_modinit_type_init_code", 0);
  /*--- Type init code ---*/
  if (PyType_Ready(&__pyx_type_14crowdposetools_5_mask_RLEs) < 0) __PYX_ERR(0, 56, __pyx_L1_error)
  __pyx_type_14crowdposetools_5_mask_RLEs.tp_print = 0;
  if (PyObject_SetAttrString(__pyx_m, "RLEs", (PyObject *)&__pyx_type_14crowdposetools_5_mask_RLEs) < 0) __PYX_ERR(0, 56, __pyx_L1_error)
  if (__Pyx_setup_reduce((PyObject*)&__pyx_type_14crowdposetools_5_mask_RLEs) < 0) __PYX_ERR(0, 56, __pyx_L1_error)
  __pyx_ptype_14crowdposetools_5_mask_RLEs = &__pyx_type_14crowdposetools_5_mask_RLEs;
  if (PyType_Ready(&__pyx_type_14crowdposetools_5_mask_Masks) < 0) __PYX_ERR(0, 77, __pyx_L1_error)
  __pyx_type_14crowdposetools_5_mask_Masks.tp_print = 0;
  if ((CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP) && likely(!__pyx_type_14crowdposetools_5_mask_Masks.tp_dictoffset && __pyx_type_14crowdposetools_5_mask_Masks.tp_getattro == PyObject_GenericGetAttr)) {
    __pyx_type_14crowdposetools_5_mask_Masks.tp_getattro = __Pyx_PyObject_GenericGetAttr;
  }
  if (PyObject_SetAttrString(__pyx_m, "Masks", (PyObject *)&__pyx_type_14crowdposetools_5_mask_Masks) < 0) __PYX_ERR(0, 77, __pyx_L1_error)
  if (__Pyx_setup_reduce((PyObject*)&__pyx_type_14crowdposetools_5_mask_Masks) < 0) __PYX_ERR(0, 77, __pyx_L1_error)
  __pyx_ptype_14crowdposetools_5_mask_Masks = &__pyx_type_14crowdposetools_5_mask_Masks;
  __Pyx_RefNannyFinishContext();
  return 0;
  __pyx_L1_error:;
  __Pyx_RefNannyFinishContext();
  return -1;
}

static int __Pyx_modinit_type_import_code(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_modinit_type_import_code", 0);
  /*--- Type import code ---*/
  __pyx_ptype_7cpython_4type_type = __Pyx_ImportType(__Pyx_BUILTIN_MODULE_NAME, "type", 
  #if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x050B0000
  sizeof(PyTypeObject),
  #else
  sizeof(PyHeapTypeObject),
  #endif
  0); if (unlikely(!__pyx_ptype_7cpython_4type_type)) __PYX_ERR(3, 9, __pyx_L1_error)
  __pyx_ptype_5numpy_dtype = __Pyx_ImportType("numpy", "dtype", sizeof(PyArray_Descr), 0); if (unlikely(!__pyx_ptype_5numpy_dtype)) __PYX_ERR(2, 164, __pyx_L1_error)
  __pyx_ptype_5numpy_flatiter = __Pyx_ImportType("numpy", "flatiter", sizeof(PyArrayIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_flatiter)) __PYX_ERR(2, 186, __pyx_L1_error)
  __pyx_ptype_5numpy_broadcast = __Pyx_ImportType("numpy", "broadcast", sizeof(PyArrayMultiIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_broadcast)) __PYX_ERR(2, 190, __pyx_L1_error)
  __pyx_ptype_5numpy_ndarray = __Pyx_ImportType("numpy", "ndarray", sizeof(PyArrayObject), 0); if (unlikely(!__pyx_ptype_5numpy_ndarray)) __PYX_ERR(2, 199, __pyx_L1_error)
  __pyx_ptype_5numpy_ufunc = __Pyx_ImportType("numpy", "ufunc", sizeof(PyUFuncObject), 0); if (unlikely(!__pyx_ptype_5numpy_ufunc)) __PYX_ERR(2, 872, __pyx_L1_error)
  __Pyx_RefNannyFinishContext();
  return 0;
  __pyx_L1_error:;
  __Pyx_RefNannyFinishContext();
  return -1;
}

static int __Pyx_modinit_variable_import_code(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_modinit_variable_import_code", 0);
  /*--- Variable import code ---*/
  __Pyx_RefNannyFinishContext();
  return 0;
}

static int __Pyx_modinit_function_import_code(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_modinit_function_import_code", 0);
  /*--- Function import code ---*/
  __Pyx_RefNannyFinishContext();
  return 0;
}


#if PY_MAJOR_VERSION < 3
#ifdef CYTHON_NO_PYINIT_EXPORT
#define __Pyx_PyMODINIT_FUNC void
#else
#define __Pyx_PyMODINIT_FUNC PyMODINIT_FUNC
#endif
#else
#ifdef CYTHON_NO_PYINIT_EXPORT
#define __Pyx_PyMODINIT_FUNC PyObject *
#else
#define __Pyx_PyMODINIT_FUNC PyMODINIT_FUNC
#endif
#endif
#ifndef CYTHON_SMALL_CODE
#if defined(__clang__)
    #define CYTHON_SMALL_CODE
#elif defined(__GNUC__) && (!(defined(__cplusplus)) || (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 4)))
    #define CYTHON_SMALL_CODE __attribute__((cold))
#else
    #define CYTHON_SMALL_CODE
#endif
#endif


#if PY_MAJOR_VERSION < 3
__Pyx_PyMODINIT_FUNC init_mask(void) CYTHON_SMALL_CODE; /*proto*/
__Pyx_PyMODINIT_FUNC init_mask(void)
#else
__Pyx_PyMODINIT_FUNC PyInit__mask(void) CYTHON_SMALL_CODE; /*proto*/
__Pyx_PyMODINIT_FUNC PyInit__mask(void)
#if CYTHON_PEP489_MULTI_PHASE_INIT
{
  return PyModuleDef_Init(&__pyx_moduledef);
}
static int __Pyx_copy_spec_to_module(PyObject *spec, PyObject *moddict, const char* from_name, const char* to_name) {
    PyObject *value = PyObject_GetAttrString(spec, from_name);
    int result = 0;
    if (likely(value)) {
        result = PyDict_SetItemString(moddict, to_name, value);
        Py_DECREF(value);
    } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
    } else {
        result = -1;
    }
    return result;
}
static PyObject* __pyx_pymod_create(PyObject *spec, CYTHON_UNUSED PyModuleDef *def) {
    PyObject *module = NULL, *moddict, *modname;
    if (__pyx_m)
        return __Pyx_NewRef(__pyx_m);
    modname = PyObject_GetAttrString(spec, "name");
    if (unlikely(!modname)) goto bad;
    module = PyModule_NewObject(modname);
    Py_DECREF(modname);
    if (unlikely(!module)) goto bad;
    moddict = PyModule_GetDict(module);
    if (unlikely(!moddict)) goto bad;
    if (unlikely(__Pyx_copy_spec_to_module(spec, moddict, "loader", "__loader__") < 0)) goto bad;
    if (unlikely(__Pyx_copy_spec_to_module(spec, moddict, "origin", "__file__") < 0)) goto bad;
    if (unlikely(__Pyx_copy_spec_to_module(spec, moddict, "parent", "__package__") < 0)) goto bad;
    if (unlikely(__Pyx_copy_spec_to_module(spec, moddict, "submodule_search_locations", "__path__") < 0)) goto bad;
    return module;
bad:
    Py_XDECREF(module);
    return NULL;
}


static int __pyx_pymod_exec__mask(PyObject *__pyx_pyinit_module)
#endif
#endif
{
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  int __pyx_t_3;
  __Pyx_RefNannyDeclarations
  #if CYTHON_PEP489_MULTI_PHASE_INIT
  if (__pyx_m && __pyx_m == __pyx_pyinit_module) return 0;
  #elif PY_MAJOR_VERSION >= 3
  if (__pyx_m) return __Pyx_NewRef(__pyx_m);
  #endif
  #if CYTHON_REFNANNY
__Pyx_RefNanny = __Pyx_RefNannyImportAPI("refnanny");
if (!__Pyx_RefNanny) {
  PyErr_Clear();
  __Pyx_RefNanny = __Pyx_RefNannyImportAPI("Cython.Runtime.refnanny");
  if (!__Pyx_RefNanny)
      Py_FatalError("failed to import 'refnanny' module");
}
#endif
  __Pyx_RefNannySetupContext("__Pyx_PyMODINIT_FUNC PyInit__mask(void)", 0);
  if (__Pyx_check_binary_version() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_tuple = PyTuple_New(0); if (unlikely(!__pyx_empty_tuple)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_bytes = PyBytes_FromStringAndSize("", 0); if (unlikely(!__pyx_empty_bytes)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_empty_unicode = PyUnicode_FromStringAndSize("", 0); if (unlikely(!__pyx_empty_unicode)) __PYX_ERR(0, 1, __pyx_L1_error)
  #ifdef __Pyx_CyFunction_USED
  if (__pyx_CyFunction_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_FusedFunction_USED
  if (__pyx_FusedFunction_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_Coroutine_USED
  if (__pyx_Coroutine_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_Generator_USED
  if (__pyx_Generator_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_AsyncGen_USED
  if (__pyx_AsyncGen_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  #ifdef __Pyx_StopAsyncIteration_USED
  if (__pyx_StopAsyncIteration_init() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  /*--- Library function declarations ---*/
  /*--- Threads initialization code ---*/
  #if defined(__PYX_FORCE_INIT_THREADS) && __PYX_FORCE_INIT_THREADS
  #ifdef WITH_THREAD /* Python build with threading support? */
  PyEval_InitThreads();
  #endif
  #endif
  /*--- Module creation code ---*/
  #if CYTHON_PEP489_MULTI_PHASE_INIT
  __pyx_m = __pyx_pyinit_module;
  Py_INCREF(__pyx_m);
  #else
  #if PY_MAJOR_VERSION < 3
  __pyx_m = Py_InitModule4("_mask", __pyx_methods, 0, 0, PYTHON_API_VERSION); Py_XINCREF(__pyx_m);
  #else
  __pyx_m = PyModule_Create(&__pyx_moduledef);
  #endif
  if (unlikely(!__pyx_m)) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  __pyx_d = PyModule_GetDict(__pyx_m); if (unlikely(!__pyx_d)) __PYX_ERR(0, 1, __pyx_L1_error)
  Py_INCREF(__pyx_d);
  __pyx_b = PyImport_AddModule(__Pyx_BUILTIN_MODULE_NAME); if (unlikely(!__pyx_b)) __PYX_ERR(0, 1, __pyx_L1_error)
  __pyx_cython_runtime = PyImport_AddModule((char *) "cython_runtime"); if (unlikely(!__pyx_cython_runtime)) __PYX_ERR(0, 1, __pyx_L1_error)
  #if CYTHON_COMPILING_IN_PYPY
  Py_INCREF(__pyx_b);
  #endif
  if (PyObject_SetAttrString(__pyx_m, "__builtins__", __pyx_b) < 0) __PYX_ERR(0, 1, __pyx_L1_error);
  /*--- Initialize various global constants etc. ---*/
  if (__Pyx_InitGlobals() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #if PY_MAJOR_VERSION < 3 && (__PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT)
  if (__Pyx_init_sys_getdefaultencoding_params() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif
  if (__pyx_module_is_main_crowdposetools___mask) {
    if (PyObject_SetAttrString(__pyx_m, "__name__", __pyx_n_s_main) < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  }
  #if PY_MAJOR_VERSION >= 3
  {
    PyObject *modules = PyImport_GetModuleDict(); if (unlikely(!modules)) __PYX_ERR(0, 1, __pyx_L1_error)
    if (!PyDict_GetItemString(modules, "crowdposetools._mask")) {
      if (unlikely(PyDict_SetItemString(modules, "crowdposetools._mask", __pyx_m) < 0)) __PYX_ERR(0, 1, __pyx_L1_error)
    }
  }
  #endif
  /*--- Builtin init code ---*/
  if (__Pyx_InitCachedBuiltins() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  /*--- Constants init code ---*/
  if (__Pyx_InitCachedConstants() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  /*--- Global type/function init code ---*/
  (void)__Pyx_modinit_global_init_code();
  (void)__Pyx_modinit_variable_export_code();
  (void)__Pyx_modinit_function_export_code();
  if (unlikely(__Pyx_modinit_type_init_code() != 0)) goto __pyx_L1_error;
  if (unlikely(__Pyx_modinit_type_import_code() != 0)) goto __pyx_L1_error;
  (void)__Pyx_modinit_variable_import_code();
  (void)__Pyx_modinit_function_import_code();
  /*--- Execution code ---*/
  #if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)
  if (__Pyx_patch_abc() < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  #endif

  /* "crowdposetools/_mask.pyx":11
 * #**************************************************************************
 * 
 * __author__ = 'tsungyi'             # <<<<<<<<<<<<<<
 * 
 * import sys
 */
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_author, __pyx_n_s_tsungyi) < 0) __PYX_ERR(0, 11, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":13
 * __author__ = 'tsungyi'
 * 
 * import sys             # <<<<<<<<<<<<<<
 * PYTHON_VERSION = sys.version_info[0]
 * 
 */
  __pyx_t_1 = __Pyx_Import(__pyx_n_s_sys, 0, -1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 13, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_sys, __pyx_t_1) < 0) __PYX_ERR(0, 13, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":14
 * 
 * import sys
 * PYTHON_VERSION = sys.version_info[0]             # <<<<<<<<<<<<<<
 * 
 * # import both Python-level and C-level symbols of Numpy
 */
  __pyx_t_1 = __Pyx_GetModuleGlobalName(__pyx_n_s_sys); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 14, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_t_1, __pyx_n_s_version_info); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 14, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = __Pyx_GetItemInt(__pyx_t_2, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 14, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_PYTHON_VERSION, __pyx_t_1) < 0) __PYX_ERR(0, 14, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":18
 * # import both Python-level and C-level symbols of Numpy
 * # the API uses Numpy to interface C and Python
 * import numpy as np             # <<<<<<<<<<<<<<
 * cimport numpy as np
 * from libc.stdlib cimport malloc, free
 */
  __pyx_t_1 = __Pyx_Import(__pyx_n_s_numpy, 0, -1); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 18, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_np, __pyx_t_1) < 0) __PYX_ERR(0, 18, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":23
 * 
 * # intialized Numpy. must do.
 * np.import_array()             # <<<<<<<<<<<<<<
 * 
 * # import numpy C function
 */
  __pyx_t_3 = __pyx_f_5numpy_import_array(); if (unlikely(__pyx_t_3 == ((int)-1))) __PYX_ERR(0, 23, __pyx_L1_error)

  /* "crowdposetools/_mask.pyx":103
 * 
 * # internal conversion from Python RLEs object to compressed RLE format
 * def _toString(RLEs Rs):             # <<<<<<<<<<<<<<
 *     cdef siz n = Rs.n
 *     cdef bytes py_string
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_1_toString, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 103, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_toString, __pyx_t_1) < 0) __PYX_ERR(0, 103, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":119
 * 
 * # internal conversion from compressed RLE format to Python RLEs object
 * def _frString(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef siz n = len(rleObjs)
 *     Rs = RLEs(n)
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_3_frString, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 119, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_frString, __pyx_t_1) < 0) __PYX_ERR(0, 119, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":137
 * # encode mask to RLEs objects
 * # list of RLE string can be generated by RLEs member function
 * def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):             # <<<<<<<<<<<<<<
 *     h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
 *     cdef RLEs Rs = RLEs(n)
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_5encode, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 137, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_encode, __pyx_t_1) < 0) __PYX_ERR(0, 137, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":145
 * 
 * # decode mask from compressed list of RLE string or RLEs object
 * def decode(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_7decode, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 145, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_decode, __pyx_t_1) < 0) __PYX_ERR(0, 145, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":152
 *     return np.array(masks)
 * 
 * def merge(rleObjs, intersect=0):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef RLEs R = RLEs(1)
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_9merge, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 152, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_merge, __pyx_t_1) < 0) __PYX_ERR(0, 152, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":159
 *     return obj
 * 
 * def area(rleObjs):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_11area, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 159, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_area, __pyx_t_1) < 0) __PYX_ERR(0, 159, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":171
 * 
 * # iou computation. support function overload (RLEs-RLEs and bbox-bbox).
 * def iou( dt, gt, pyiscrowd ):             # <<<<<<<<<<<<<<
 *     def _preproc(objs):
 *         if len(objs) == 0:
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_13iou, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 171, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_iou_2, __pyx_t_1) < 0) __PYX_ERR(0, 171, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":241
 *     return iou.reshape((m,n), order='F')
 * 
 * def toBbox( rleObjs ):             # <<<<<<<<<<<<<<
 *     cdef RLEs Rs = _frString(rleObjs)
 *     cdef siz n = Rs.n
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_15toBbox, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 241, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_toBbox, __pyx_t_1) < 0) __PYX_ERR(0, 241, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":253
 *     return bb
 * 
 * def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef siz n = bb.shape[0]
 *     Rs = RLEs(n)
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_17frBbox, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 253, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_frBbox, __pyx_t_1) < 0) __PYX_ERR(0, 253, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":260
 *     return objs
 * 
 * def frPoly( poly, siz h, siz w ):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.double_t, ndim=1] np_poly
 *     n = len(poly)
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_19frPoly, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 260, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_frPoly, __pyx_t_1) < 0) __PYX_ERR(0, 260, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":270
 *     return objs
 * 
 * def frUncompressedRLE(ucRles, siz h, siz w):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.uint32_t, ndim=1] cnts
 *     cdef RLE R
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_21frUncompressedRLE, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 270, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_frUncompressedRLE, __pyx_t_1) < 0) __PYX_ERR(0, 270, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":288
 *     return objs
 * 
 * def frPyObjects(pyobj, h, w):             # <<<<<<<<<<<<<<
 *     # encode rle from a list of python objects
 *     if type(pyobj) == np.ndarray:
 */
  __pyx_t_1 = PyCFunction_NewEx(&__pyx_mdef_14crowdposetools_5_mask_23frPyObjects, NULL, __pyx_n_s_crowdposetools__mask); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 288, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_frPyObjects, __pyx_t_1) < 0) __PYX_ERR(0, 288, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "crowdposetools/_mask.pyx":1
 * # distutils: language = c             # <<<<<<<<<<<<<<
 * # distutils: sources = ../common/maskApi.c
 * 
 */
  __pyx_t_1 = __Pyx_PyDict_NewPresized(0); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 1, __pyx_L1_error)
  __Pyx_GOTREF(__pyx_t_1);
  if (PyDict_SetItem(__pyx_d, __pyx_n_s_test, __pyx_t_1) < 0) __PYX_ERR(0, 1, __pyx_L1_error)
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "../../../anaconda3/envs/py0/lib/python3.6/site-packages/Cython/Includes/numpy/__init__.pxd":1008
 *         raise ImportError("numpy.core.umath failed to import")
 * 
 * cdef inline int import_ufunc() except -1:             # <<<<<<<<<<<<<<
 *     try:
 *         _import_umath()
 */

  /*--- Wrapped vars code ---*/

  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  if (__pyx_m) {
    if (__pyx_d) {
      __Pyx_AddTraceback("init crowdposetools._mask", 0, __pyx_lineno, __pyx_filename);
    }
    Py_DECREF(__pyx_m); __pyx_m = 0;
  } else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_ImportError, "init crowdposetools._mask");
  }
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  #if CYTHON_PEP489_MULTI_PHASE_INIT
  return (__pyx_m != NULL) ? 0 : -1;
  #elif PY_MAJOR_VERSION >= 3
  return __pyx_m;
  #else
  return;
  #endif
}

/* --- Runtime support code --- */
/* Refnanny */
#if CYTHON_REFNANNY
static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname) {
    PyObject *m = NULL, *p = NULL;
    void *r = NULL;
    m = PyImport_ImportModule((char *)modname);
    if (!m) goto end;
    p = PyObject_GetAttrString(m, (char *)"RefNannyAPI");
    if (!p) goto end;
    r = PyLong_AsVoidPtr(p);
end:
    Py_XDECREF(p);
    Py_XDECREF(m);
    return (__Pyx_RefNannyAPIStruct *)r;
}
#endif

/* PyObjectGetAttrStr */
#if CYTHON_USE_TYPE_SLOTS
static CYTHON_INLINE PyObject* __Pyx_PyObject_GetAttrStr(PyObject* obj, PyObject* attr_name) {
    PyTypeObject* tp = Py_TYPE(obj);
    if (likely(tp->tp_getattro))
        return tp->tp_getattro(obj, attr_name);
#if PY_MAJOR_VERSION < 3
    if (likely(tp->tp_getattr))
        return tp->tp_getattr(obj, PyString_AS_STRING(attr_name));
#endif
    return PyObject_GetAttr(obj, attr_name);
}
#endif

/* GetBuiltinName */
static PyObject *__Pyx_GetBuiltinName(PyObject *name) {
    PyObject* result = __Pyx_PyObject_GetAttrStr(__pyx_b, name);
    if (unlikely(!result)) {
        PyErr_Format(PyExc_NameError,
#if PY_MAJOR_VERSION >= 3
            "name '%U' is not defined", name);
#else
            "name '%.200s' is not defined", PyString_AS_STRING(name));
#endif
    }
    return result;
}

/* RaiseDoubleKeywords */
static void __Pyx_RaiseDoubleKeywordsError(
    const char* func_name,
    PyObject* kw_name)
{
    PyErr_Format(PyExc_TypeError,
        #if PY_MAJOR_VERSION >= 3
        "%s() got multiple values for keyword argument '%U'", func_name, kw_name);
        #else
        "%s() got multiple values for keyword argument '%s'", func_name,
        PyString_AsString(kw_name));
        #endif
}

/* ParseKeywords */
static int __Pyx_ParseOptionalKeywords(
    PyObject *kwds,
    PyObject **argnames[],
    PyObject *kwds2,
    PyObject *values[],
    Py_ssize_t num_pos_args,
    const char* function_name)
{
    PyObject *key = 0, *value = 0;
    Py_ssize_t pos = 0;
    PyObject*** name;
    PyObject*** first_kw_arg = argnames + num_pos_args;
    while (PyDict_Next(kwds, &pos, &key, &value)) {
        name = first_kw_arg;
        while (*name && (**name != key)) name++;
        if (*name) {
            values[name-argnames] = value;
            continue;
        }
        name = first_kw_arg;
        #if PY_MAJOR_VERSION < 3
        if (likely(PyString_CheckExact(key)) || likely(PyString_Check(key))) {
            while (*name) {
                if ((CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**name) == PyString_GET_SIZE(key))
                        && _PyString_Eq(**name, key)) {
                    values[name-argnames] = value;
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    if ((**argname == key) || (
                            (CYTHON_COMPILING_IN_PYPY || PyString_GET_SIZE(**argname) == PyString_GET_SIZE(key))
                             && _PyString_Eq(**argname, key))) {
                        goto arg_passed_twice;
                    }
                    argname++;
                }
            }
        } else
        #endif
        if (likely(PyUnicode_Check(key))) {
            while (*name) {
                int cmp = (**name == key) ? 0 :
                #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                    (PyUnicode_GET_SIZE(**name) != PyUnicode_GET_SIZE(key)) ? 1 :
                #endif
                    PyUnicode_Compare(**name, key);
                if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                if (cmp == 0) {
                    values[name-argnames] = value;
                    break;
                }
                name++;
            }
            if (*name) continue;
            else {
                PyObject*** argname = argnames;
                while (argname != first_kw_arg) {
                    int cmp = (**argname == key) ? 0 :
                    #if !CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION >= 3
                        (PyUnicode_GET_SIZE(**argname) != PyUnicode_GET_SIZE(key)) ? 1 :
                    #endif
                        PyUnicode_Compare(**argname, key);
                    if (cmp < 0 && unlikely(PyErr_Occurred())) goto bad;
                    if (cmp == 0) goto arg_passed_twice;
                    argname++;
                }
            }
        } else
            goto invalid_keyword_type;
        if (kwds2) {
            if (unlikely(PyDict_SetItem(kwds2, key, value))) goto bad;
        } else {
            goto invalid_keyword;
        }
    }
    return 0;
arg_passed_twice:
    __Pyx_RaiseDoubleKeywordsError(function_name, key);
    goto bad;
invalid_keyword_type:
    PyErr_Format(PyExc_TypeError,
        "%.200s() keywords must be strings", function_name);
    goto bad;
invalid_keyword:
    PyErr_Format(PyExc_TypeError,
    #if PY_MAJOR_VERSION < 3
        "%.200s() got an unexpected keyword argument '%.200s'",
        function_name, PyString_AsString(key));
    #else
        "%s() got an unexpected keyword argument '%U'",
        function_name, key);
    #endif
bad:
    return -1;
}

/* RaiseArgTupleInvalid */
static void __Pyx_RaiseArgtupleInvalid(
    const char* func_name,
    int exact,
    Py_ssize_t num_min,
    Py_ssize_t num_max,
    Py_ssize_t num_found)
{
    Py_ssize_t num_expected;
    const char *more_or_less;
    if (num_found < num_min) {
        num_expected = num_min;
        more_or_less = "at least";
    } else {
        num_expected = num_max;
        more_or_less = "at most";
    }
    if (exact) {
        more_or_less = "exactly";
    }
    PyErr_Format(PyExc_TypeError,
                 "%.200s() takes %.8s %" CYTHON_FORMAT_SSIZE_T "d positional argument%.1s (%" CYTHON_FORMAT_SSIZE_T "d given)",
                 func_name, more_or_less, num_expected,
                 (num_expected == 1) ? "" : "s", num_found);
}

/* BytesEquals */
static CYTHON_INLINE int __Pyx_PyBytes_Equals(PyObject* s1, PyObject* s2, int equals) {
#if CYTHON_COMPILING_IN_PYPY
    return PyObject_RichCompareBool(s1, s2, equals);
#else
    if (s1 == s2) {
        return (equals == Py_EQ);
    } else if (PyBytes_CheckExact(s1) & PyBytes_CheckExact(s2)) {
        const char *ps1, *ps2;
        Py_ssize_t length = PyBytes_GET_SIZE(s1);
        if (length != PyBytes_GET_SIZE(s2))
            return (equals == Py_NE);
        ps1 = PyBytes_AS_STRING(s1);
        ps2 = PyBytes_AS_STRING(s2);
        if (ps1[0] != ps2[0]) {
            return (equals == Py_NE);
        } else if (length == 1) {
            return (equals == Py_EQ);
        } else {
            int result;
#if CYTHON_USE_UNICODE_INTERNALS
            Py_hash_t hash1, hash2;
            hash1 = ((PyBytesObject*)s1)->ob_shash;
            hash2 = ((PyBytesObject*)s2)->ob_shash;
            if (hash1 != hash2 && hash1 != -1 && hash2 != -1) {
                return (equals == Py_NE);
            }
#endif
            result = memcmp(ps1, ps2, (size_t)length);
            return (equals == Py_EQ) ? (result == 0) : (result != 0);
        }
    } else if ((s1 == Py_None) & PyBytes_CheckExact(s2)) {
        return (equals == Py_NE);
    } else if ((s2 == Py_None) & PyBytes_CheckExact(s1)) {
        return (equals == Py_NE);
    } else {
        int result;
        PyObject* py_result = PyObject_RichCompare(s1, s2, equals);
        if (!py_result)
            return -1;
        result = __Pyx_PyObject_IsTrue(py_result);
        Py_DECREF(py_result);
        return result;
    }
#endif
}

/* UnicodeEquals */
static CYTHON_INLINE int __Pyx_PyUnicode_Equals(PyObject* s1, PyObject* s2, int equals) {
#if CYTHON_COMPILING_IN_PYPY
    return PyObject_RichCompareBool(s1, s2, equals);
#else
#if PY_MAJOR_VERSION < 3
    PyObject* owned_ref = NULL;
#endif
    int s1_is_unicode, s2_is_unicode;
    if (s1 == s2) {
        goto return_eq;
    }
    s1_is_unicode = PyUnicode_CheckExact(s1);
    s2_is_unicode = PyUnicode_CheckExact(s2);
#if PY_MAJOR_VERSION < 3
    if ((s1_is_unicode & (!s2_is_unicode)) && PyString_CheckExact(s2)) {
        owned_ref = PyUnicode_FromObject(s2);
        if (unlikely(!owned_ref))
            return -1;
        s2 = owned_ref;
        s2_is_unicode = 1;
    } else if ((s2_is_unicode & (!s1_is_unicode)) && PyString_CheckExact(s1)) {
        owned_ref = PyUnicode_FromObject(s1);
        if (unlikely(!owned_ref))
            return -1;
        s1 = owned_ref;
        s1_is_unicode = 1;
    } else if (((!s2_is_unicode) & (!s1_is_unicode))) {
        return __Pyx_PyBytes_Equals(s1, s2, equals);
    }
#endif
    if (s1_is_unicode & s2_is_unicode) {
        Py_ssize_t length;
        int kind;
        void *data1, *data2;
        if (unlikely(__Pyx_PyUnicode_READY(s1) < 0) || unlikely(__Pyx_PyUnicode_READY(s2) < 0))
            return -1;
        length = __Pyx_PyUnicode_GET_LENGTH(s1);
        if (length != __Pyx_PyUnicode_GET_LENGTH(s2)) {
            goto return_ne;
        }
#if CYTHON_USE_UNICODE_INTERNALS
        {
            Py_hash_t hash1, hash2;
        #if CYTHON_PEP393_ENABLED
            hash1 = ((PyASCIIObject*)s1)->hash;
            hash2 = ((PyASCIIObject*)s2)->hash;
        #else
            hash1 = ((PyUnicodeObject*)s1)->hash;
            hash2 = ((PyUnicodeObject*)s2)->hash;
        #endif
            if (hash1 != hash2 && hash1 != -1 && hash2 != -1) {
                goto return_ne;
            }
        }
#endif
        kind = __Pyx_PyUnicode_KIND(s1);
        if (kind != __Pyx_PyUnicode_KIND(s2)) {
            goto return_ne;
        }
        data1 = __Pyx_PyUnicode_DATA(s1);
        data2 = __Pyx_PyUnicode_DATA(s2);
        if (__Pyx_PyUnicode_READ(kind, data1, 0) != __Pyx_PyUnicode_READ(kind, data2, 0)) {
            goto return_ne;
        } else if (length == 1) {
            goto return_eq;
        } else {
            int result = memcmp(data1, data2, (size_t)(length * kind));
            #if PY_MAJOR_VERSION < 3
            Py_XDECREF(owned_ref);
            #endif
            return (equals == Py_EQ) ? (result == 0) : (result != 0);
        }
    } else if ((s1 == Py_None) & s2_is_unicode) {
        goto return_ne;
    } else if ((s2 == Py_None) & s1_is_unicode) {
        goto return_ne;
    } else {
        int result;
        PyObject* py_result = PyObject_RichCompare(s1, s2, equals);
        #if PY_MAJOR_VERSION < 3
        Py_XDECREF(owned_ref);
        #endif
        if (!py_result)
            return -1;
        result = __Pyx_PyObject_IsTrue(py_result);
        Py_DECREF(py_result);
        return result;
    }
return_eq:
    #if PY_MAJOR_VERSION < 3
    Py_XDECREF(owned_ref);
    #endif
    return (equals == Py_EQ);
return_ne:
    #if PY_MAJOR_VERSION < 3
    Py_XDECREF(owned_ref);
    #endif
    return (equals == Py_NE);
#endif
}

/* PyCFunctionFastCall */
#if CYTHON_FAST_PYCCALL
static CYTHON_INLINE PyObject * __Pyx_PyCFunction_FastCall(PyObject *func_obj, PyObject **args, Py_ssize_t nargs) {
    PyCFunctionObject *func = (PyCFunctionObject*)func_obj;
    PyCFunction meth = PyCFunction_GET_FUNCTION(func);
    PyObject *self = PyCFunction_GET_SELF(func);
    int flags = PyCFunction_GET_FLAGS(func);
    assert(PyCFunction_Check(func));
    assert(METH_FASTCALL == (flags & ~(METH_CLASS | METH_STATIC | METH_COEXIST | METH_KEYWORDS)));
    assert(nargs >= 0);
    assert(nargs == 0 || args != NULL);
    /* _PyCFunction_FastCallDict() must not be called with an exception set,
       because it may clear it (directly or indirectly) and so the
       caller loses its exception */
    assert(!PyErr_Occurred());
    if ((PY_VERSION_HEX < 0x030700A0) || unlikely(flags & METH_KEYWORDS)) {
        return (*((__Pyx_PyCFunctionFastWithKeywords)meth)) (self, args, nargs, NULL);
    } else {
        return (*((__Pyx_PyCFunctionFast)meth)) (self, args, nargs);
    }
}
#endif

/* PyFunctionFastCall */
#if CYTHON_FAST_PYCALL
#include "frameobject.h"
static PyObject* __Pyx_PyFunction_FastCallNoKw(PyCodeObject *co, PyObject **args, Py_ssize_t na,
                                               PyObject *globals) {
    PyFrameObject *f;
    PyThreadState *tstate = __Pyx_PyThreadState_Current;
    PyObject **fastlocals;
    Py_ssize_t i;
    PyObject *result;
    assert(globals != NULL);
    /* XXX Perhaps we should create a specialized
       PyFrame_New() that doesn't take locals, but does
       take builtins without sanity checking them.
       */
    assert(tstate != NULL);
    f = PyFrame_New(tstate, co, globals, NULL);
    if (f == NULL) {
        return NULL;
    }
    fastlocals = f->f_localsplus;
    for (i = 0; i < na; i++) {
        Py_INCREF(*args);
        fastlocals[i] = *args++;
    }
    result = PyEval_EvalFrameEx(f,0);
    ++tstate->recursion_depth;
    Py_DECREF(f);
    --tstate->recursion_depth;
    return result;
}
#if 1 || PY_VERSION_HEX < 0x030600B1
static PyObject *__Pyx_PyFunction_FastCallDict(PyObject *func, PyObject **args, int nargs, PyObject *kwargs) {
    PyCodeObject *co = (PyCodeObject *)PyFunction_GET_CODE(func);
    PyObject *globals = PyFunction_GET_GLOBALS(func);
    PyObject *argdefs = PyFunction_GET_DEFAULTS(func);
    PyObject *closure;
#if PY_MAJOR_VERSION >= 3
    PyObject *kwdefs;
#endif
    PyObject *kwtuple, **k;
    PyObject **d;
    Py_ssize_t nd;
    Py_ssize_t nk;
    PyObject *result;
    assert(kwargs == NULL || PyDict_Check(kwargs));
    nk = kwargs ? PyDict_Size(kwargs) : 0;
    if (Py_EnterRecursiveCall((char*)" while calling a Python object")) {
        return NULL;
    }
    if (
#if PY_MAJOR_VERSION >= 3
            co->co_kwonlyargcount == 0 &&
#endif
            likely(kwargs == NULL || nk == 0) &&
            co->co_flags == (CO_OPTIMIZED | CO_NEWLOCALS | CO_NOFREE)) {
        if (argdefs == NULL && co->co_argcount == nargs) {
            result = __Pyx_PyFunction_FastCallNoKw(co, args, nargs, globals);
            goto done;
        }
        else if (nargs == 0 && argdefs != NULL
                 && co->co_argcount == Py_SIZE(argdefs)) {
            /* function called with no arguments, but all parameters have
               a default value: use default values as arguments .*/
            args = &PyTuple_GET_ITEM(argdefs, 0);
            result =__Pyx_PyFunction_FastCallNoKw(co, args, Py_SIZE(argdefs), globals);
            goto done;
        }
    }
    if (kwargs != NULL) {
        Py_ssize_t pos, i;
        kwtuple = PyTuple_New(2 * nk);
        if (kwtuple == NULL) {
            result = NULL;
            goto done;
        }
        k = &PyTuple_GET_ITEM(kwtuple, 0);
        pos = i = 0;
        while (PyDict_Next(kwargs, &pos, &k[i], &k[i+1])) {
            Py_INCREF(k[i]);
            Py_INCREF(k[i+1]);
            i += 2;
        }
        nk = i / 2;
    }
    else {
        kwtuple = NULL;
        k = NULL;
    }
    closure = PyFunction_GET_CLOSURE(func);
#if PY_MAJOR_VERSION >= 3
    kwdefs = PyFunction_GET_KW_DEFAULTS(func);
#endif
    if (argdefs != NULL) {
        d = &PyTuple_GET_ITEM(argdefs, 0);
        nd = Py_SIZE(argdefs);
    }
    else {
        d = NULL;
        nd = 0;
    }
#if PY_MAJOR_VERSION >= 3
    result = PyEval_EvalCodeEx((PyObject*)co, globals, (PyObject *)NULL,
                               args, nargs,
                               k, (int)nk,
                               d, (int)nd, kwdefs, closure);
#else
    result = PyEval_EvalCodeEx(co, globals, (PyObject *)NULL,
                               args, nargs,
                               k, (int)nk,
                               d, (int)nd, closure);
#endif
    Py_XDECREF(kwtuple);
done:
    Py_LeaveRecursiveCall();
    return result;
}
#endif
#endif

/* PyObjectCall */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    PyObject *result;
    ternaryfunc call = func->ob_type->tp_call;
    if (unlikely(!call))
        return PyObject_Call(func, arg, kw);
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object")))
        return NULL;
    result = (*call)(func, arg, kw);
    Py_LeaveRecursiveCall();
    if (unlikely(!result) && unlikely(!PyErr_Occurred())) {
        PyErr_SetString(
            PyExc_SystemError,
            "NULL result without error in PyObject_Call");
    }
    return result;
}
#endif

/* PyObjectCallMethO */
#if CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallMethO(PyObject *func, PyObject *arg) {
    PyObject *self, *result;
    PyCFunction cfunc;
    cfunc = PyCFunction_GET_FUNCTION(func);
    self = PyCFunction_GET_SELF(func);
    if (unlikely(Py_EnterRecursiveCall((char*)" while calling a Python object")))
        return NULL;
    result = cfunc(self, arg);
    Py_LeaveRecursiveCall();
    if (unlikely(!result) && unlikely(!PyErr_Occurred())) {
        PyErr_SetString(
            PyExc_SystemError,
            "NULL result without error in PyObject_Call");
    }
    return result;
}
#endif

/* PyObjectCallOneArg */
#if CYTHON_COMPILING_IN_CPYTHON
static PyObject* __Pyx__PyObject_CallOneArg(PyObject *func, PyObject *arg) {
    PyObject *result;
    PyObject *args = PyTuple_New(1);
    if (unlikely(!args)) return NULL;
    Py_INCREF(arg);
    PyTuple_SET_ITEM(args, 0, arg);
    result = __Pyx_PyObject_Call(func, args, NULL);
    Py_DECREF(args);
    return result;
}
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg) {
#if CYTHON_FAST_PYCALL
    if (PyFunction_Check(func)) {
        return __Pyx_PyFunction_FastCall(func, &arg, 1);
    }
#endif
    if (likely(PyCFunction_Check(func))) {
        if (likely(PyCFunction_GET_FLAGS(func) & METH_O)) {
            return __Pyx_PyObject_CallMethO(func, arg);
#if CYTHON_FAST_PYCCALL
        } else if (PyCFunction_GET_FLAGS(func) & METH_FASTCALL) {
            return __Pyx_PyCFunction_FastCall(func, &arg, 1);
#endif
        }
    }
    return __Pyx__PyObject_CallOneArg(func, arg);
}
#else
static CYTHON_INLINE PyObject* __Pyx_PyObject_CallOneArg(PyObject *func, PyObject *arg) {
    PyObject *result;
    PyObject *args = PyTuple_Pack(1, arg);
    if (unlikely(!args)) return NULL;
    result = __Pyx_PyObject_Call(func, args, NULL);
    Py_DECREF(args);
    return result;
}
#endif

/* PyErrFetchRestore */
#if CYTHON_FAST_THREAD_STATE
static CYTHON_INLINE void __Pyx_ErrRestoreInState(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb) {
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    tmp_type = tstate->curexc_type;
    tmp_value = tstate->curexc_value;
    tmp_tb = tstate->curexc_traceback;
    tstate->curexc_type = type;
    tstate->curexc_value = value;
    tstate->curexc_traceback = tb;
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
}
static CYTHON_INLINE void __Pyx_ErrFetchInState(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb) {
    *type = tstate->curexc_type;
    *value = tstate->curexc_value;
    *tb = tstate->curexc_traceback;
    tstate->curexc_type = 0;
    tstate->curexc_value = 0;
    tstate->curexc_traceback = 0;
}
#endif

/* RaiseException */
#if PY_MAJOR_VERSION < 3
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb,
                        CYTHON_UNUSED PyObject *cause) {
    __Pyx_PyThreadState_declare
    Py_XINCREF(type);
    if (!value || value == Py_None)
        value = NULL;
    else
        Py_INCREF(value);
    if (!tb || tb == Py_None)
        tb = NULL;
    else {
        Py_INCREF(tb);
        if (!PyTraceBack_Check(tb)) {
            PyErr_SetString(PyExc_TypeError,
                "raise: arg 3 must be a traceback or None");
            goto raise_error;
        }
    }
    if (PyType_Check(type)) {
#if CYTHON_COMPILING_IN_PYPY
        if (!value) {
            Py_INCREF(Py_None);
            value = Py_None;
        }
#endif
        PyErr_NormalizeException(&type, &value, &tb);
    } else {
        if (value) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto raise_error;
        }
        value = type;
        type = (PyObject*) Py_TYPE(type);
        Py_INCREF(type);
        if (!PyType_IsSubtype((PyTypeObject *)type, (PyTypeObject *)PyExc_BaseException)) {
            PyErr_SetString(PyExc_TypeError,
                "raise: exception class must be a subclass of BaseException");
            goto raise_error;
        }
    }
    __Pyx_PyThreadState_assign
    __Pyx_ErrRestore(type, value, tb);
    return;
raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    Py_XDECREF(tb);
    return;
}
#else
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause) {
    PyObject* owned_instance = NULL;
    if (tb == Py_None) {
        tb = 0;
    } else if (tb && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: arg 3 must be a traceback or None");
        goto bad;
    }
    if (value == Py_None)
        value = 0;
    if (PyExceptionInstance_Check(type)) {
        if (value) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto bad;
        }
        value = type;
        type = (PyObject*) Py_TYPE(value);
    } else if (PyExceptionClass_Check(type)) {
        PyObject *instance_class = NULL;
        if (value && PyExceptionInstance_Check(value)) {
            instance_class = (PyObject*) Py_TYPE(value);
            if (instance_class != type) {
                int is_subclass = PyObject_IsSubclass(instance_class, type);
                if (!is_subclass) {
                    instance_class = NULL;
                } else if (unlikely(is_subclass == -1)) {
                    goto bad;
                } else {
                    type = instance_class;
                }
            }
        }
        if (!instance_class) {
            PyObject *args;
            if (!value)
                args = PyTuple_New(0);
            else if (PyTuple_Check(value)) {
                Py_INCREF(value);
                args = value;
            } else
                args = PyTuple_Pack(1, value);
            if (!args)
                goto bad;
            owned_instance = PyObject_Call(type, args, NULL);
            Py_DECREF(args);
            if (!owned_instance)
                goto bad;
            value = owned_instance;
            if (!PyExceptionInstance_Check(value)) {
                PyErr_Format(PyExc_TypeError,
                             "calling %R should have returned an instance of "
                             "BaseException, not %R",
                             type, Py_TYPE(value));
                goto bad;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
            "raise: exception class must be a subclass of BaseException");
        goto bad;
    }
    if (cause) {
        PyObject *fixed_cause;
        if (cause == Py_None) {
            fixed_cause = NULL;
        } else if (PyExceptionClass_Check(cause)) {
            fixed_cause = PyObject_CallObject(cause, NULL);
            if (fixed_cause == NULL)
                goto bad;
        } else if (PyExceptionInstance_Check(cause)) {
            fixed_cause = cause;
            Py_INCREF(fixed_cause);
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "exception causes must derive from "
                            "BaseException");
            goto bad;
        }
        PyException_SetCause(value, fixed_cause);
    }
    PyErr_SetObject(type, value);
    if (tb) {
#if CYTHON_COMPILING_IN_PYPY
        PyObject *tmp_type, *tmp_value, *tmp_tb;
        PyErr_Fetch(&tmp_type, &tmp_value, &tmp_tb);
        Py_INCREF(tb);
        PyErr_Restore(tmp_type, tmp_value, tb);
        Py_XDECREF(tmp_tb);
#else
        PyThreadState *tstate = __Pyx_PyThreadState_Current;
        PyObject* tmp_tb = tstate->curexc_traceback;
        if (tb != tmp_tb) {
            Py_INCREF(tb);
            tstate->curexc_traceback = tb;
            Py_XDECREF(tmp_tb);
        }
#endif
    }
bad:
    Py_XDECREF(owned_instance);
    return;
}
#endif

/* ExtTypeTest */
static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type) {
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (likely(__Pyx_TypeCheck(obj, type)))
        return 1;
    PyErr_Format(PyExc_TypeError, "Cannot convert %.200s to %.200s",
                 Py_TYPE(obj)->tp_name, type->tp_name);
    return 0;
}

/* ArgTypeTest */
static int __Pyx__ArgTypeTest(PyObject *obj, PyTypeObject *type, const char *name, int exact)
{
    if (unlikely(!type)) {
        PyErr_SetString(PyExc_SystemError, "Missing type object");
        return 0;
    }
    else if (exact) {
        #if PY_MAJOR_VERSION == 2
        if ((type == &PyBaseString_Type) && likely(__Pyx_PyBaseString_CheckExact(obj))) return 1;
        #endif
    }
    else {
        if (likely(__Pyx_TypeCheck(obj, type))) return 1;
    }
    PyErr_Format(PyExc_TypeError,
        "Argument '%.200s' has incorrect type (expected %.200s, got %.200s)",
        name, type->tp_name, Py_TYPE(obj)->tp_name);
    return 0;
}

/* PyIntBinop */
#if !CYTHON_COMPILING_IN_PYPY
static PyObject* __Pyx_PyInt_AddObjC(PyObject *op1, PyObject *op2, CYTHON_UNUSED long intval, CYTHON_UNUSED int inplace) {
    #if PY_MAJOR_VERSION < 3
    if (likely(PyInt_CheckExact(op1))) {
        const long b = intval;
        long x;
        long a = PyInt_AS_LONG(op1);
            x = (long)((unsigned long)a + b);
            if (likely((x^a) >= 0 || (x^b) >= 0))
                return PyInt_FromLong(x);
            return PyLong_Type.tp_as_number->nb_add(op1, op2);
    }
    #endif
    #if CYTHON_USE_PYLONG_INTERNALS
    if (likely(PyLong_CheckExact(op1))) {
        const long b = intval;
        long a, x;
#ifdef HAVE_LONG_LONG
        const PY_LONG_LONG llb = intval;
        PY_LONG_LONG lla, llx;
#endif
        const digit* digits = ((PyLongObject*)op1)->ob_digit;
        const Py_ssize_t size = Py_SIZE(op1);
        if (likely(__Pyx_sst_abs(size) <= 1)) {
            a = likely(size) ? digits[0] : 0;
            if (size == -1) a = -a;
        } else {
            switch (size) {
                case -2:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        a = -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 2 * PyLong_SHIFT) {
                        lla = -(PY_LONG_LONG) (((((unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                    CYTHON_FALLTHROUGH;
                case 2:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        a = (long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 2 * PyLong_SHIFT) {
                        lla = (PY_LONG_LONG) (((((unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                    CYTHON_FALLTHROUGH;
                case -3:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        a = -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 3 * PyLong_SHIFT) {
                        lla = -(PY_LONG_LONG) (((((((unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                    CYTHON_FALLTHROUGH;
                case 3:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        a = (long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 3 * PyLong_SHIFT) {
                        lla = (PY_LONG_LONG) (((((((unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                    CYTHON_FALLTHROUGH;
                case -4:
                    if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                        a = -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 4 * PyLong_SHIFT) {
                        lla = -(PY_LONG_LONG) (((((((((unsigned PY_LONG_LONG)digits[3]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                    CYTHON_FALLTHROUGH;
                case 4:
                    if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                        a = (long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > 4 * PyLong_SHIFT) {
                        lla = (PY_LONG_LONG) (((((((((unsigned PY_LONG_LONG)digits[3]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[2]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[1]) << PyLong_SHIFT) | (unsigned PY_LONG_LONG)digits[0]));
                        goto long_long;
#endif
                    }
                    CYTHON_FALLTHROUGH;
                default: return PyLong_Type.tp_as_number->nb_add(op1, op2);
            }
        }
                x = a + b;
            return PyLong_FromLong(x);
#ifdef HAVE_LONG_LONG
        long_long:
                llx = lla + llb;
            return PyLong_FromLongLong(llx);
#endif
        
        
    }
    #endif
    if (PyFloat_CheckExact(op1)) {
        const long b = intval;
        double a = PyFloat_AS_DOUBLE(op1);
            double result;
            PyFPE_START_PROTECT("add", return NULL)
            result = ((double)a) + (double)b;
            PyFPE_END_PROTECT(result)
            return PyFloat_FromDouble(result);
    }
    return (inplace ? PyNumber_InPlaceAdd : PyNumber_Add)(op1, op2);
}
#endif

/* PyIntBinop */
#if !CYTHON_COMPILING_IN_PYPY
static PyObject* __Pyx_PyInt_EqObjC(PyObject *op1, PyObject *op2, CYTHON_UNUSED long intval, CYTHON_UNUSED int inplace) {
    if (op1 == op2) {
        Py_RETURN_TRUE;
    }
    #if PY_MAJOR_VERSION < 3
    if (likely(PyInt_CheckExact(op1))) {
        const long b = intval;
        long a = PyInt_AS_LONG(op1);
        if (a == b) {
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }
    }
    #endif
    #if CYTHON_USE_PYLONG_INTERNALS
    if (likely(PyLong_CheckExact(op1))) {
        const long b = intval;
        long a;
        const digit* digits = ((PyLongObject*)op1)->ob_digit;
        const Py_ssize_t size = Py_SIZE(op1);
        if (likely(__Pyx_sst_abs(size) <= 1)) {
            a = likely(size) ? digits[0] : 0;
            if (size == -1) a = -a;
        } else {
            switch (size) {
                case -2:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        a = -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
                    }
                    CYTHON_FALLTHROUGH;
                case 2:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        a = (long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
                    }
                    CYTHON_FALLTHROUGH;
                case -3:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        a = -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
                    }
                    CYTHON_FALLTHROUGH;
                case 3:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        a = (long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
                    }
                    CYTHON_FALLTHROUGH;
                case -4:
                    if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                        a = -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
                    }
                    CYTHON_FALLTHROUGH;
                case 4:
                    if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                        a = (long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0]));
                        break;
                    }
                    CYTHON_FALLTHROUGH;
                #if PyLong_SHIFT < 30 && PyLong_SHIFT != 15
                default: return PyLong_Type.tp_richcompare(op1, op2, Py_EQ);
                #else
                default: Py_RETURN_FALSE;
                #endif
            }
        }
            if (a == b) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
    }
    #endif
    if (PyFloat_CheckExact(op1)) {
        const long b = intval;
        double a = PyFloat_AS_DOUBLE(op1);
            if ((double)a == (double)b) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
    }
    return PyObject_RichCompare(op1, op2, Py_EQ);
}
#endif

/* GetModuleGlobalName */
static CYTHON_INLINE PyObject *__Pyx_GetModuleGlobalName(PyObject *name) {
    PyObject *result;
#if !CYTHON_AVOID_BORROWED_REFS
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030500A1
    result = _PyDict_GetItem_KnownHash(__pyx_d, name, ((PyASCIIObject *) name)->hash);
    if (likely(result)) {
        Py_INCREF(result);
    } else if (unlikely(PyErr_Occurred())) {
        result = NULL;
    } else {
#else
    result = PyDict_GetItem(__pyx_d, name);
    if (likely(result)) {
        Py_INCREF(result);
    } else {
#endif
#else
    result = PyObject_GetItem(__pyx_d, name);
    if (!result) {
        PyErr_Clear();
#endif
        result = __Pyx_GetBuiltinName(name);
    }
    return result;
}

/* DictGetItem */
    #if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
static PyObject *__Pyx_PyDict_GetItem(PyObject *d, PyObject* key) {
    PyObject *value;
    value = PyDict_GetItemWithError(d, key);
    if (unlikely(!value)) {
        if (!PyErr_Occurred()) {
            PyObject* args = PyTuple_Pack(1, key);
            if (likely(args))
                PyErr_SetObject(PyExc_KeyError, args);
            Py_XDECREF(args);
        }
        return NULL;
    }
    Py_INCREF(value);
    return value;
}
#endif

/* GetItemInt */
    static PyObject *__Pyx_GetItemInt_Generic(PyObject *o, PyObject* j) {
    PyObject *r;
    if (!j) return NULL;
    r = PyObject_GetItem(o, j);
    Py_DECREF(j);
    return r;
}
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_List_Fast(PyObject *o, Py_ssize_t i,
                                                              CYTHON_NCP_UNUSED int wraparound,
                                                              CYTHON_NCP_UNUSED int boundscheck) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    Py_ssize_t wrapped_i = i;
    if (wraparound & unlikely(i < 0)) {
        wrapped_i += PyList_GET_SIZE(o);
    }
    if ((!boundscheck) || likely((0 <= wrapped_i) & (wrapped_i < PyList_GET_SIZE(o)))) {
        PyObject *r = PyList_GET_ITEM(o, wrapped_i);
        Py_INCREF(r);
        return r;
    }
    return __Pyx_GetItemInt_Generic(o, PyInt_FromSsize_t(i));
#else
    return PySequence_GetItem(o, i);
#endif
}
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_Tuple_Fast(PyObject *o, Py_ssize_t i,
                                                              CYTHON_NCP_UNUSED int wraparound,
                                                              CYTHON_NCP_UNUSED int boundscheck) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    Py_ssize_t wrapped_i = i;
    if (wraparound & unlikely(i < 0)) {
        wrapped_i += PyTuple_GET_SIZE(o);
    }
    if ((!boundscheck) || likely((0 <= wrapped_i) & (wrapped_i < PyTuple_GET_SIZE(o)))) {
        PyObject *r = PyTuple_GET_ITEM(o, wrapped_i);
        Py_INCREF(r);
        return r;
    }
    return __Pyx_GetItemInt_Generic(o, PyInt_FromSsize_t(i));
#else
    return PySequence_GetItem(o, i);
#endif
}
static CYTHON_INLINE PyObject *__Pyx_GetItemInt_Fast(PyObject *o, Py_ssize_t i, int is_list,
                                                     CYTHON_NCP_UNUSED int wraparound,
                                                     CYTHON_NCP_UNUSED int boundscheck) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS && CYTHON_USE_TYPE_SLOTS
    if (is_list || PyList_CheckExact(o)) {
        Py_ssize_t n = ((!wraparound) | likely(i >= 0)) ? i : i + PyList_GET_SIZE(o);
        if ((!boundscheck) || (likely((n >= 0) & (n < PyList_GET_SIZE(o))))) {
            PyObject *r = PyList_GET_ITEM(o, n);
            Py_INCREF(r);
            return r;
        }
    }
    else if (PyTuple_CheckExact(o)) {
        Py_ssize_t n = ((!wraparound) | likely(i >= 0)) ? i : i + PyTuple_GET_SIZE(o);
        if ((!boundscheck) || likely((n >= 0) & (n < PyTuple_GET_SIZE(o)))) {
            PyObject *r = PyTuple_GET_ITEM(o, n);
            Py_INCREF(r);
            return r;
        }
    } else {
        PySequenceMethods *m = Py_TYPE(o)->tp_as_sequence;
        if (likely(m && m->sq_item)) {
            if (wraparound && unlikely(i < 0) && likely(m->sq_length)) {
                Py_ssize_t l = m->sq_length(o);
                if (likely(l >= 0)) {
                    i += l;
                } else {
                    if (!PyErr_ExceptionMatches(PyExc_OverflowError))
                        return NULL;
                    PyErr_Clear();
                }
            }
            return m->sq_item(o, i);
        }
    }
#else
    if (is_list || PySequence_Check(o)) {
        return PySequence_GetItem(o, i);
    }
#endif
    return __Pyx_GetItemInt_Generic(o, PyInt_FromSsize_t(i));
}

/* IsLittleEndian */
    static CYTHON_INLINE int __Pyx_Is_Little_Endian(void)
{
  union {
    uint32_t u32;
    uint8_t u8[4];
  } S;
  S.u32 = 0x01020304;
  return S.u8[0] == 4;
}

/* BufferFormatCheck */
    static void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type) {
  stack[0].field = &ctx->root;
  stack[0].parent_offset = 0;
  ctx->root.type = type;
  ctx->root.name = "buffer dtype";
  ctx->root.offset = 0;
  ctx->head = stack;
  ctx->head->field = &ctx->root;
  ctx->fmt_offset = 0;
  ctx->head->parent_offset = 0;
  ctx->new_packmode = '@';
  ctx->enc_packmode = '@';
  ctx->new_count = 1;
  ctx->enc_count = 0;
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  ctx->is_valid_array = 0;
  ctx->struct_alignment = 0;
  while (type->typegroup == 'S') {
    ++ctx->head;
    ctx->head->field = type->fields;
    ctx->head->parent_offset = 0;
    type = type->fields->type;
  }
}
static int __Pyx_BufFmt_ParseNumber(const char** ts) {
    int count;
    const char* t = *ts;
    if (*t < '0' || *t > '9') {
      return -1;
    } else {
        count = *t++ - '0';
        while (*t >= '0' && *t < '9') {
            count *= 10;
            count += *t++ - '0';
        }
    }
    *ts = t;
    return count;
}
static int __Pyx_BufFmt_ExpectNumber(const char **ts) {
    int number = __Pyx_BufFmt_ParseNumber(ts);
    if (number == -1)
        PyErr_Format(PyExc_ValueError,\
                     "Does not understand character buffer dtype format string ('%c')", **ts);
    return number;
}
static void __Pyx_BufFmt_RaiseUnexpectedChar(char ch) {
  PyErr_Format(PyExc_ValueError,
               "Unexpected format string character: '%c'", ch);
}
static const char* __Pyx_BufFmt_DescribeTypeChar(char ch, int is_complex) {
  switch (ch) {
    case 'c': return "'char'";
    case 'b': return "'signed char'";
    case 'B': return "'unsigned char'";
    case 'h': return "'short'";
    case 'H': return "'unsigned short'";
    case 'i': return "'int'";
    case 'I': return "'unsigned int'";
    case 'l': return "'long'";
    case 'L': return "'unsigned long'";
    case 'q': return "'long long'";
    case 'Q': return "'unsigned long long'";
    case 'f': return (is_complex ? "'complex float'" : "'float'");
    case 'd': return (is_complex ? "'complex double'" : "'double'");
    case 'g': return (is_complex ? "'complex long double'" : "'long double'");
    case 'T': return "a struct";
    case 'O': return "Python object";
    case 'P': return "a pointer";
    case 's': case 'p': return "a string";
    case 0: return "end";
    default: return "unparseable format string";
  }
}
static size_t __Pyx_BufFmt_TypeCharToStandardSize(char ch, int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return 2;
    case 'i': case 'I': case 'l': case 'L': return 4;
    case 'q': case 'Q': return 8;
    case 'f': return (is_complex ? 8 : 4);
    case 'd': return (is_complex ? 16 : 8);
    case 'g': {
      PyErr_SetString(PyExc_ValueError, "Python does not define a standard format string size for long double ('g')..");
      return 0;
    }
    case 'O': case 'P': return sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
static size_t __Pyx_BufFmt_TypeCharToNativeSize(char ch, int is_complex) {
  switch (ch) {
    case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(short);
    case 'i': case 'I': return sizeof(int);
    case 'l': case 'L': return sizeof(long);
    #ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(PY_LONG_LONG);
    #endif
    case 'f': return sizeof(float) * (is_complex ? 2 : 1);
    case 'd': return sizeof(double) * (is_complex ? 2 : 1);
    case 'g': return sizeof(long double) * (is_complex ? 2 : 1);
    case 'O': case 'P': return sizeof(void*);
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}
typedef struct { char c; short x; } __Pyx_st_short;
typedef struct { char c; int x; } __Pyx_st_int;
typedef struct { char c; long x; } __Pyx_st_long;
typedef struct { char c; float x; } __Pyx_st_float;
typedef struct { char c; double x; } __Pyx_st_double;
typedef struct { char c; long double x; } __Pyx_st_longdouble;
typedef struct { char c; void *x; } __Pyx_st_void_p;
#ifdef HAVE_LONG_LONG
typedef struct { char c; PY_LONG_LONG x; } __Pyx_st_longlong;
#endif
static size_t __Pyx_BufFmt_TypeCharToAlignment(char ch, CYTHON_UNUSED int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(__Pyx_st_short) - sizeof(short);
    case 'i': case 'I': return sizeof(__Pyx_st_int) - sizeof(int);
    case 'l': case 'L': return sizeof(__Pyx_st_long) - sizeof(long);
#ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(__Pyx_st_longlong) - sizeof(PY_LONG_LONG);
#endif
    case 'f': return sizeof(__Pyx_st_float) - sizeof(float);
    case 'd': return sizeof(__Pyx_st_double) - sizeof(double);
    case 'g': return sizeof(__Pyx_st_longdouble) - sizeof(long double);
    case 'P': case 'O': return sizeof(__Pyx_st_void_p) - sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
/* These are for computing the padding at the end of the struct to align
   on the first member of the struct. This will probably the same as above,
   but we don't have any guarantees.
 */
typedef struct { short x; char c; } __Pyx_pad_short;
typedef struct { int x; char c; } __Pyx_pad_int;
typedef struct { long x; char c; } __Pyx_pad_long;
typedef struct { float x; char c; } __Pyx_pad_float;
typedef struct { double x; char c; } __Pyx_pad_double;
typedef struct { long double x; char c; } __Pyx_pad_longdouble;
typedef struct { void *x; char c; } __Pyx_pad_void_p;
#ifdef HAVE_LONG_LONG
typedef struct { PY_LONG_LONG x; char c; } __Pyx_pad_longlong;
#endif
static size_t __Pyx_BufFmt_TypeCharToPadding(char ch, CYTHON_UNUSED int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': case 's': case 'p': return 1;
    case 'h': case 'H': return sizeof(__Pyx_pad_short) - sizeof(short);
    case 'i': case 'I': return sizeof(__Pyx_pad_int) - sizeof(int);
    case 'l': case 'L': return sizeof(__Pyx_pad_long) - sizeof(long);
#ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(__Pyx_pad_longlong) - sizeof(PY_LONG_LONG);
#endif
    case 'f': return sizeof(__Pyx_pad_float) - sizeof(float);
    case 'd': return sizeof(__Pyx_pad_double) - sizeof(double);
    case 'g': return sizeof(__Pyx_pad_longdouble) - sizeof(long double);
    case 'P': case 'O': return sizeof(__Pyx_pad_void_p) - sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}
static char __Pyx_BufFmt_TypeCharToGroup(char ch, int is_complex) {
  switch (ch) {
    case 'c':
        return 'H';
    case 'b': case 'h': case 'i':
    case 'l': case 'q': case 's': case 'p':
        return 'I';
    case 'B': case 'H': case 'I': case 'L': case 'Q':
        return 'U';
    case 'f': case 'd': case 'g':
        return (is_complex ? 'C' : 'R');
    case 'O':
        return 'O';
    case 'P':
        return 'P';
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}
static void __Pyx_BufFmt_RaiseExpected(__Pyx_BufFmt_Context* ctx) {
  if (ctx->head == NULL || ctx->head->field == &ctx->root) {
    const char* expected;
    const char* quote;
    if (ctx->head == NULL) {
      expected = "end";
      quote = "";
    } else {
      expected = ctx->head->field->type->name;
      quote = "'";
    }
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected %s%s%s but got %s",
                 quote, expected, quote,
                 __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex));
  } else {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_StructField* parent = (ctx->head - 1)->field;
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected '%s' but got %s in '%s.%s'",
                 field->type->name, __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex),
                 parent->type->name, field->name);
  }
}
static int __Pyx_BufFmt_ProcessTypeChunk(__Pyx_BufFmt_Context* ctx) {
  char group;
  size_t size, offset, arraysize = 1;
  if (ctx->enc_type == 0) return 0;
  if (ctx->head->field->type->arraysize[0]) {
    int i, ndim = 0;
    if (ctx->enc_type == 's' || ctx->enc_type == 'p') {
        ctx->is_valid_array = ctx->head->field->type->ndim == 1;
        ndim = 1;
        if (ctx->enc_count != ctx->head->field->type->arraysize[0]) {
            PyErr_Format(PyExc_ValueError,
                         "Expected a dimension of size %zu, got %zu",
                         ctx->head->field->type->arraysize[0], ctx->enc_count);
            return -1;
        }
    }
    if (!ctx->is_valid_array) {
      PyErr_Format(PyExc_ValueError, "Expected %d dimensions, got %d",
                   ctx->head->field->type->ndim, ndim);
      return -1;
    }
    for (i = 0; i < ctx->head->field->type->ndim; i++) {
      arraysize *= ctx->head->field->type->arraysize[i];
    }
    ctx->is_valid_array = 0;
    ctx->enc_count = 1;
  }
  group = __Pyx_BufFmt_TypeCharToGroup(ctx->enc_type, ctx->is_complex);
  do {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_TypeInfo* type = field->type;
    if (ctx->enc_packmode == '@' || ctx->enc_packmode == '^') {
      size = __Pyx_BufFmt_TypeCharToNativeSize(ctx->enc_type, ctx->is_complex);
    } else {
      size = __Pyx_BufFmt_TypeCharToStandardSize(ctx->enc_type, ctx->is_complex);
    }
    if (ctx->enc_packmode == '@') {
      size_t align_at = __Pyx_BufFmt_TypeCharToAlignment(ctx->enc_type, ctx->is_complex);
      size_t align_mod_offset;
      if (align_at == 0) return -1;
      align_mod_offset = ctx->fmt_offset % align_at;
      if (align_mod_offset > 0) ctx->fmt_offset += align_at - align_mod_offset;
      if (ctx->struct_alignment == 0)
          ctx->struct_alignment = __Pyx_BufFmt_TypeCharToPadding(ctx->enc_type,
                                                                 ctx->is_complex);
    }
    if (type->size != size || type->typegroup != group) {
      if (type->typegroup == 'C' && type->fields != NULL) {
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        ++ctx->head;
        ctx->head->field = type->fields;
        ctx->head->parent_offset = parent_offset;
        continue;
      }
      if ((type->typegroup == 'H' || group == 'H') && type->size == size) {
      } else {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return -1;
      }
    }
    offset = ctx->head->parent_offset + field->offset;
    if (ctx->fmt_offset != offset) {
      PyErr_Format(PyExc_ValueError,
                   "Buffer dtype mismatch; next field is at offset %" CYTHON_FORMAT_SSIZE_T "d but %" CYTHON_FORMAT_SSIZE_T "d expected",
                   (Py_ssize_t)ctx->fmt_offset, (Py_ssize_t)offset);
      return -1;
    }
    ctx->fmt_offset += size;
    if (arraysize)
      ctx->fmt_offset += (arraysize - 1) * size;
    --ctx->enc_count;
    while (1) {
      if (field == &ctx->root) {
        ctx->head = NULL;
        if (ctx->enc_count != 0) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return -1;
        }
        break;
      }
      ctx->head->field = ++field;
      if (field->type == NULL) {
        --ctx->head;
        field = ctx->head->field;
        continue;
      } else if (field->type->typegroup == 'S') {
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        if (field->type->fields->type == NULL) continue;
        field = field->type->fields;
        ++ctx->head;
        ctx->head->field = field;
        ctx->head->parent_offset = parent_offset;
        break;
      } else {
        break;
      }
    }
  } while (ctx->enc_count);
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  return 0;
}
static PyObject *
__pyx_buffmt_parse_array(__Pyx_BufFmt_Context* ctx, const char** tsp)
{
    const char *ts = *tsp;
    int i = 0, number;
    int ndim = ctx->head->field->type->ndim;
;
    ++ts;
    if (ctx->new_count != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot handle repeated arrays in format string");
        return NULL;
    }
    if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
    while (*ts && *ts != ')') {
        switch (*ts) {
            case ' ': case '\f': case '\r': case '\n': case '\t': case '\v':  continue;
            default:  break;
        }
        number = __Pyx_BufFmt_ExpectNumber(&ts);
        if (number == -1) return NULL;
        if (i < ndim && (size_t) number != ctx->head->field->type->arraysize[i])
            return PyErr_Format(PyExc_ValueError,
                        "Expected a dimension of size %zu, got %d",
                        ctx->head->field->type->arraysize[i], number);
        if (*ts != ',' && *ts != ')')
            return PyErr_Format(PyExc_ValueError,
                                "Expected a comma in format string, got '%c'", *ts);
        if (*ts == ',') ts++;
        i++;
    }
    if (i != ndim)
        return PyErr_Format(PyExc_ValueError, "Expected %d dimension(s), got %d",
                            ctx->head->field->type->ndim, i);
    if (!*ts) {
        PyErr_SetString(PyExc_ValueError,
                        "Unexpected end of format string, expected ')'");
        return NULL;
    }
    ctx->is_valid_array = 1;
    ctx->new_count = 1;
    *tsp = ++ts;
    return Py_None;
}
static const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts) {
  int got_Z = 0;
  while (1) {
    switch(*ts) {
      case 0:
        if (ctx->enc_type != 0 && ctx->head == NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        if (ctx->head != NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        return ts;
      case ' ':
      case '\r':
      case '\n':
        ++ts;
        break;
      case '<':
        if (!__Pyx_Is_Little_Endian()) {
          PyErr_SetString(PyExc_ValueError, "Little-endian buffer not supported on big-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '>':
      case '!':
        if (__Pyx_Is_Little_Endian()) {
          PyErr_SetString(PyExc_ValueError, "Big-endian buffer not supported on little-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '=':
      case '@':
      case '^':
        ctx->new_packmode = *ts++;
        break;
      case 'T':
        {
          const char* ts_after_sub;
          size_t i, struct_count = ctx->new_count;
          size_t struct_alignment = ctx->struct_alignment;
          ctx->new_count = 1;
          ++ts;
          if (*ts != '{') {
            PyErr_SetString(PyExc_ValueError, "Buffer acquisition: Expected '{' after 'T'");
            return NULL;
          }
          if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
          ctx->enc_type = 0;
          ctx->enc_count = 0;
          ctx->struct_alignment = 0;
          ++ts;
          ts_after_sub = ts;
          for (i = 0; i != struct_count; ++i) {
            ts_after_sub = __Pyx_BufFmt_CheckString(ctx, ts);
            if (!ts_after_sub) return NULL;
          }
          ts = ts_after_sub;
          if (struct_alignment) ctx->struct_alignment = struct_alignment;
        }
        break;
      case '}':
        {
          size_t alignment = ctx->struct_alignment;
          ++ts;
          if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
          ctx->enc_type = 0;
          if (alignment && ctx->fmt_offset % alignment) {
            ctx->fmt_offset += alignment - (ctx->fmt_offset % alignment);
          }
        }
        return ts;
      case 'x':
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        ctx->fmt_offset += ctx->new_count;
        ctx->new_count = 1;
        ctx->enc_count = 0;
        ctx->enc_type = 0;
        ctx->enc_packmode = ctx->new_packmode;
        ++ts;
        break;
      case 'Z':
        got_Z = 1;
        ++ts;
        if (*ts != 'f' && *ts != 'd' && *ts != 'g') {
          __Pyx_BufFmt_RaiseUnexpectedChar('Z');
          return NULL;
        }
        CYTHON_FALLTHROUGH;
      case 'c': case 'b': case 'B': case 'h': case 'H': case 'i': case 'I':
      case 'l': case 'L': case 'q': case 'Q':
      case 'f': case 'd': case 'g':
      case 'O': case 'p':
        if (ctx->enc_type == *ts && got_Z == ctx->is_complex &&
            ctx->enc_packmode == ctx->new_packmode) {
          ctx->enc_count += ctx->new_count;
          ctx->new_count = 1;
          got_Z = 0;
          ++ts;
          break;
        }
        CYTHON_FALLTHROUGH;
      case 's':
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        ctx->enc_count = ctx->new_count;
        ctx->enc_packmode = ctx->new_packmode;
        ctx->enc_type = *ts;
        ctx->is_complex = got_Z;
        ++ts;
        ctx->new_count = 1;
        got_Z = 0;
        break;
      case ':':
        ++ts;
        while(*ts != ':') ++ts;
        ++ts;
        break;
      case '(':
        if (!__pyx_buffmt_parse_array(ctx, &ts)) return NULL;
        break;
      default:
        {
          int number = __Pyx_BufFmt_ExpectNumber(&ts);
          if (number == -1) return NULL;
          ctx->new_count = (size_t)number;
        }
    }
  }
}

/* BufferGetAndValidate */
      static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info) {
  if (unlikely(info->buf == NULL)) return;
  if (info->suboffsets == __Pyx_minusones) info->suboffsets = NULL;
  __Pyx_ReleaseBuffer(info);
}
static void __Pyx_ZeroBuffer(Py_buffer* buf) {
  buf->buf = NULL;
  buf->obj = NULL;
  buf->strides = __Pyx_zeros;
  buf->shape = __Pyx_zeros;
  buf->suboffsets = __Pyx_minusones;
}
static int __Pyx__GetBufferAndValidate(
        Py_buffer* buf, PyObject* obj,  __Pyx_TypeInfo* dtype, int flags,
        int nd, int cast, __Pyx_BufFmt_StackElem* stack)
{
  buf->buf = NULL;
  if (unlikely(__Pyx_GetBuffer(obj, buf, flags) == -1)) {
    __Pyx_ZeroBuffer(buf);
    return -1;
  }
  if (unlikely(buf->ndim != nd)) {
    PyErr_Format(PyExc_ValueError,
                 "Buffer has wrong number of dimensions (expected %d, got %d)",
                 nd, buf->ndim);
    goto fail;
  }
  if (!cast) {
    __Pyx_BufFmt_Context ctx;
    __Pyx_BufFmt_Init(&ctx, stack, dtype);
    if (!__Pyx_BufFmt_CheckString(&ctx, buf->format)) goto fail;
  }
  if (unlikely((unsigned)buf->itemsize != dtype->size)) {
    PyErr_Format(PyExc_ValueError,
      "Item size of buffer (%" CYTHON_FORMAT_SSIZE_T "d byte%s) does not match size of '%s' (%" CYTHON_FORMAT_SSIZE_T "d byte%s)",
      buf->itemsize, (buf->itemsize > 1) ? "s" : "",
      dtype->name, (Py_ssize_t)dtype->size, (dtype->size > 1) ? "s" : "");
    goto fail;
  }
  if (buf->suboffsets == NULL) buf->suboffsets = __Pyx_minusones;
  return 0;
fail:;
  __Pyx_SafeReleaseBuffer(buf);
  return -1;
}

/* FetchCommonType */
      static PyTypeObject* __Pyx_FetchCommonType(PyTypeObject* type) {
    PyObject* fake_module;
    PyTypeObject* cached_type = NULL;
    fake_module = PyImport_AddModule((char*) "_cython_" CYTHON_ABI);
    if (!fake_module) return NULL;
    Py_INCREF(fake_module);
    cached_type = (PyTypeObject*) PyObject_GetAttrString(fake_module, type->tp_name);
    if (cached_type) {
        if (!PyType_Check((PyObject*)cached_type)) {
            PyErr_Format(PyExc_TypeError,
                "Shared Cython type %.200s is not a type object",
                type->tp_name);
            goto bad;
        }
        if (cached_type->tp_basicsize != type->tp_basicsize) {
            PyErr_Format(PyExc_TypeError,
                "Shared Cython type %.200s has the wrong size, try recompiling",
                type->tp_name);
            goto bad;
        }
    } else {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError)) goto bad;
        PyErr_Clear();
        if (PyType_Ready(type) < 0) goto bad;
        if (PyObject_SetAttrString(fake_module, type->tp_name, (PyObject*) type) < 0)
            goto bad;
        Py_INCREF(type);
        cached_type = type;
    }
done:
    Py_DECREF(fake_module);
    return cached_type;
bad:
    Py_XDECREF(cached_type);
    cached_type = NULL;
    goto done;
}

/* CythonFunction */
      #include <structmember.h>
static PyObject *
__Pyx_CyFunction_get_doc(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *closure)
{
    if (unlikely(op->func_doc == NULL)) {
        if (op->func.m_ml->ml_doc) {
#if PY_MAJOR_VERSION >= 3
            op->func_doc = PyUnicode_FromString(op->func.m_ml->ml_doc);
#else
            op->func_doc = PyString_FromString(op->func.m_ml->ml_doc);
#endif
            if (unlikely(op->func_doc == NULL))
                return NULL;
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
    Py_INCREF(op->func_doc);
    return op->func_doc;
}
static int
__Pyx_CyFunction_set_doc(__pyx_CyFunctionObject *op, PyObject *value)
{
    PyObject *tmp = op->func_doc;
    if (value == NULL) {
        value = Py_None;
    }
    Py_INCREF(value);
    op->func_doc = value;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject *
__Pyx_CyFunction_get_name(__pyx_CyFunctionObject *op)
{
    if (unlikely(op->func_name == NULL)) {
#if PY_MAJOR_VERSION >= 3
        op->func_name = PyUnicode_InternFromString(op->func.m_ml->ml_name);
#else
        op->func_name = PyString_InternFromString(op->func.m_ml->ml_name);
#endif
        if (unlikely(op->func_name == NULL))
            return NULL;
    }
    Py_INCREF(op->func_name);
    return op->func_name;
}
static int
__Pyx_CyFunction_set_name(__pyx_CyFunctionObject *op, PyObject *value)
{
    PyObject *tmp;
#if PY_MAJOR_VERSION >= 3
    if (unlikely(value == NULL || !PyUnicode_Check(value))) {
#else
    if (unlikely(value == NULL || !PyString_Check(value))) {
#endif
        PyErr_SetString(PyExc_TypeError,
                        "__name__ must be set to a string object");
        return -1;
    }
    tmp = op->func_name;
    Py_INCREF(value);
    op->func_name = value;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject *
__Pyx_CyFunction_get_qualname(__pyx_CyFunctionObject *op)
{
    Py_INCREF(op->func_qualname);
    return op->func_qualname;
}
static int
__Pyx_CyFunction_set_qualname(__pyx_CyFunctionObject *op, PyObject *value)
{
    PyObject *tmp;
#if PY_MAJOR_VERSION >= 3
    if (unlikely(value == NULL || !PyUnicode_Check(value))) {
#else
    if (unlikely(value == NULL || !PyString_Check(value))) {
#endif
        PyErr_SetString(PyExc_TypeError,
                        "__qualname__ must be set to a string object");
        return -1;
    }
    tmp = op->func_qualname;
    Py_INCREF(value);
    op->func_qualname = value;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject *
__Pyx_CyFunction_get_self(__pyx_CyFunctionObject *m, CYTHON_UNUSED void *closure)
{
    PyObject *self;
    self = m->func_closure;
    if (self == NULL)
        self = Py_None;
    Py_INCREF(self);
    return self;
}
static PyObject *
__Pyx_CyFunction_get_dict(__pyx_CyFunctionObject *op)
{
    if (unlikely(op->func_dict == NULL)) {
        op->func_dict = PyDict_New();
        if (unlikely(op->func_dict == NULL))
            return NULL;
    }
    Py_INCREF(op->func_dict);
    return op->func_dict;
}
static int
__Pyx_CyFunction_set_dict(__pyx_CyFunctionObject *op, PyObject *value)
{
    PyObject *tmp;
    if (unlikely(value == NULL)) {
        PyErr_SetString(PyExc_TypeError,
               "function's dictionary may not be deleted");
        return -1;
    }
    if (unlikely(!PyDict_Check(value))) {
        PyErr_SetString(PyExc_TypeError,
               "setting function's dictionary to a non-dict");
        return -1;
    }
    tmp = op->func_dict;
    Py_INCREF(value);
    op->func_dict = value;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject *
__Pyx_CyFunction_get_globals(__pyx_CyFunctionObject *op)
{
    Py_INCREF(op->func_globals);
    return op->func_globals;
}
static PyObject *
__Pyx_CyFunction_get_closure(CYTHON_UNUSED __pyx_CyFunctionObject *op)
{
    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject *
__Pyx_CyFunction_get_code(__pyx_CyFunctionObject *op)
{
    PyObject* result = (op->func_code) ? op->func_code : Py_None;
    Py_INCREF(result);
    return result;
}
static int
__Pyx_CyFunction_init_defaults(__pyx_CyFunctionObject *op) {
    int result = 0;
    PyObject *res = op->defaults_getter((PyObject *) op);
    if (unlikely(!res))
        return -1;
    #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    op->defaults_tuple = PyTuple_GET_ITEM(res, 0);
    Py_INCREF(op->defaults_tuple);
    op->defaults_kwdict = PyTuple_GET_ITEM(res, 1);
    Py_INCREF(op->defaults_kwdict);
    #else
    op->defaults_tuple = PySequence_ITEM(res, 0);
    if (unlikely(!op->defaults_tuple)) result = -1;
    else {
        op->defaults_kwdict = PySequence_ITEM(res, 1);
        if (unlikely(!op->defaults_kwdict)) result = -1;
    }
    #endif
    Py_DECREF(res);
    return result;
}
static int
__Pyx_CyFunction_set_defaults(__pyx_CyFunctionObject *op, PyObject* value) {
    PyObject* tmp;
    if (!value) {
        value = Py_None;
    } else if (value != Py_None && !PyTuple_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "__defaults__ must be set to a tuple object");
        return -1;
    }
    Py_INCREF(value);
    tmp = op->defaults_tuple;
    op->defaults_tuple = value;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject *
__Pyx_CyFunction_get_defaults(__pyx_CyFunctionObject *op) {
    PyObject* result = op->defaults_tuple;
    if (unlikely(!result)) {
        if (op->defaults_getter) {
            if (__Pyx_CyFunction_init_defaults(op) < 0) return NULL;
            result = op->defaults_tuple;
        } else {
            result = Py_None;
        }
    }
    Py_INCREF(result);
    return result;
}
static int
__Pyx_CyFunction_set_kwdefaults(__pyx_CyFunctionObject *op, PyObject* value) {
    PyObject* tmp;
    if (!value) {
        value = Py_None;
    } else if (value != Py_None && !PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "__kwdefaults__ must be set to a dict object");
        return -1;
    }
    Py_INCREF(value);
    tmp = op->defaults_kwdict;
    op->defaults_kwdict = value;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject *
__Pyx_CyFunction_get_kwdefaults(__pyx_CyFunctionObject *op) {
    PyObject* result = op->defaults_kwdict;
    if (unlikely(!result)) {
        if (op->defaults_getter) {
            if (__Pyx_CyFunction_init_defaults(op) < 0) return NULL;
            result = op->defaults_kwdict;
        } else {
            result = Py_None;
        }
    }
    Py_INCREF(result);
    return result;
}
static int
__Pyx_CyFunction_set_annotations(__pyx_CyFunctionObject *op, PyObject* value) {
    PyObject* tmp;
    if (!value || value == Py_None) {
        value = NULL;
    } else if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "__annotations__ must be set to a dict object");
        return -1;
    }
    Py_XINCREF(value);
    tmp = op->func_annotations;
    op->func_annotations = value;
    Py_XDECREF(tmp);
    return 0;
}
static PyObject *
__Pyx_CyFunction_get_annotations(__pyx_CyFunctionObject *op) {
    PyObject* result = op->func_annotations;
    if (unlikely(!result)) {
        result = PyDict_New();
        if (unlikely(!result)) return NULL;
        op->func_annotations = result;
    }
    Py_INCREF(result);
    return result;
}
static PyGetSetDef __pyx_CyFunction_getsets[] = {
    {(char *) "func_doc", (getter)__Pyx_CyFunction_get_doc, (setter)__Pyx_CyFunction_set_doc, 0, 0},
    {(char *) "__doc__",  (getter)__Pyx_CyFunction_get_doc, (setter)__Pyx_CyFunction_set_doc, 0, 0},
    {(char *) "func_name", (getter)__Pyx_CyFunction_get_name, (setter)__Pyx_CyFunction_set_name, 0, 0},
    {(char *) "__name__", (getter)__Pyx_CyFunction_get_name, (setter)__Pyx_CyFunction_set_name, 0, 0},
    {(char *) "__qualname__", (getter)__Pyx_CyFunction_get_qualname, (setter)__Pyx_CyFunction_set_qualname, 0, 0},
    {(char *) "__self__", (getter)__Pyx_CyFunction_get_self, 0, 0, 0},
    {(char *) "func_dict", (getter)__Pyx_CyFunction_get_dict, (setter)__Pyx_CyFunction_set_dict, 0, 0},
    {(char *) "__dict__", (getter)__Pyx_CyFunction_get_dict, (setter)__Pyx_CyFunction_set_dict, 0, 0},
    {(char *) "func_globals", (getter)__Pyx_CyFunction_get_globals, 0, 0, 0},
    {(char *) "__globals__", (getter)__Pyx_CyFunction_get_globals, 0, 0, 0},
    {(char *) "func_closure", (getter)__Pyx_CyFunction_get_closure, 0, 0, 0},
    {(char *) "__closure__", (getter)__Pyx_CyFunction_get_closure, 0, 0, 0},
    {(char *) "func_code", (getter)__Pyx_CyFunction_get_code, 0, 0, 0},
    {(char *) "__code__", (getter)__Pyx_CyFunction_get_code, 0, 0, 0},
    {(char *) "func_defaults", (getter)__Pyx_CyFunction_get_defaults, (setter)__Pyx_CyFunction_set_defaults, 0, 0},
    {(char *) "__defaults__", (getter)__Pyx_CyFunction_get_defaults, (setter)__Pyx_CyFunction_set_defaults, 0, 0},
    {(char *) "__kwdefaults__", (getter)__Pyx_CyFunction_get_kwdefaults, (setter)__Pyx_CyFunction_set_kwdefaults, 0, 0},
    {(char *) "__annotations__", (getter)__Pyx_CyFunction_get_annotations, (setter)__Pyx_CyFunction_set_annotations, 0, 0},
    {0, 0, 0, 0, 0}
};
static PyMemberDef __pyx_CyFunction_members[] = {
    {(char *) "__module__", T_OBJECT, offsetof(PyCFunctionObject, m_module), PY_WRITE_RESTRICTED, 0},
    {0, 0, 0,  0, 0}
};
static PyObject *
__Pyx_CyFunction_reduce(__pyx_CyFunctionObject *m, CYTHON_UNUSED PyObject *args)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromString(m->func.m_ml->ml_name);
#else
    return PyString_FromString(m->func.m_ml->ml_name);
#endif
}
static PyMethodDef __pyx_CyFunction_methods[] = {
    {"__reduce__", (PyCFunction)__Pyx_CyFunction_reduce, METH_VARARGS, 0},
    {0, 0, 0, 0}
};
#if PY_VERSION_HEX < 0x030500A0
#define __Pyx_CyFunction_weakreflist(cyfunc) ((cyfunc)->func_weakreflist)
#else
#define __Pyx_CyFunction_weakreflist(cyfunc) ((cyfunc)->func.m_weakreflist)
#endif
static PyObject *__Pyx_CyFunction_New(PyTypeObject *type, PyMethodDef *ml, int flags, PyObject* qualname,
                                      PyObject *closure, PyObject *module, PyObject* globals, PyObject* code) {
    __pyx_CyFunctionObject *op = PyObject_GC_New(__pyx_CyFunctionObject, type);
    if (op == NULL)
        return NULL;
    op->flags = flags;
    __Pyx_CyFunction_weakreflist(op) = NULL;
    op->func.m_ml = ml;
    op->func.m_self = (PyObject *) op;
    Py_XINCREF(closure);
    op->func_closure = closure;
    Py_XINCREF(module);
    op->func.m_module = module;
    op->func_dict = NULL;
    op->func_name = NULL;
    Py_INCREF(qualname);
    op->func_qualname = qualname;
    op->func_doc = NULL;
    op->func_classobj = NULL;
    op->func_globals = globals;
    Py_INCREF(op->func_globals);
    Py_XINCREF(code);
    op->func_code = code;
    op->defaults_pyobjects = 0;
    op->defaults = NULL;
    op->defaults_tuple = NULL;
    op->defaults_kwdict = NULL;
    op->defaults_getter = NULL;
    op->func_annotations = NULL;
    PyObject_GC_Track(op);
    return (PyObject *) op;
}
static int
__Pyx_CyFunction_clear(__pyx_CyFunctionObject *m)
{
    Py_CLEAR(m->func_closure);
    Py_CLEAR(m->func.m_module);
    Py_CLEAR(m->func_dict);
    Py_CLEAR(m->func_name);
    Py_CLEAR(m->func_qualname);
    Py_CLEAR(m->func_doc);
    Py_CLEAR(m->func_globals);
    Py_CLEAR(m->func_code);
    Py_CLEAR(m->func_classobj);
    Py_CLEAR(m->defaults_tuple);
    Py_CLEAR(m->defaults_kwdict);
    Py_CLEAR(m->func_annotations);
    if (m->defaults) {
        PyObject **pydefaults = __Pyx_CyFunction_Defaults(PyObject *, m);
        int i;
        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_XDECREF(pydefaults[i]);
        PyObject_Free(m->defaults);
        m->defaults = NULL;
    }
    return 0;
}
static void __Pyx__CyFunction_dealloc(__pyx_CyFunctionObject *m)
{
    if (__Pyx_CyFunction_weakreflist(m) != NULL)
        PyObject_ClearWeakRefs((PyObject *) m);
    __Pyx_CyFunction_clear(m);
    PyObject_GC_Del(m);
}
static void __Pyx_CyFunction_dealloc(__pyx_CyFunctionObject *m)
{
    PyObject_GC_UnTrack(m);
    __Pyx__CyFunction_dealloc(m);
}
static int __Pyx_CyFunction_traverse(__pyx_CyFunctionObject *m, visitproc visit, void *arg)
{
    Py_VISIT(m->func_closure);
    Py_VISIT(m->func.m_module);
    Py_VISIT(m->func_dict);
    Py_VISIT(m->func_name);
    Py_VISIT(m->func_qualname);
    Py_VISIT(m->func_doc);
    Py_VISIT(m->func_globals);
    Py_VISIT(m->func_code);
    Py_VISIT(m->func_classobj);
    Py_VISIT(m->defaults_tuple);
    Py_VISIT(m->defaults_kwdict);
    if (m->defaults) {
        PyObject **pydefaults = __Pyx_CyFunction_Defaults(PyObject *, m);
        int i;
        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_VISIT(pydefaults[i]);
    }
    return 0;
}
static PyObject *__Pyx_CyFunction_descr_get(PyObject *func, PyObject *obj, PyObject *type)
{
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    if (m->flags & __Pyx_CYFUNCTION_STATICMETHOD) {
        Py_INCREF(func);
        return func;
    }
    if (m->flags & __Pyx_CYFUNCTION_CLASSMETHOD) {
        if (type == NULL)
            type = (PyObject *)(Py_TYPE(obj));
        return __Pyx_PyMethod_New(func, type, (PyObject *)(Py_TYPE(type)));
    }
    if (obj == Py_None)
        obj = NULL;
    return __Pyx_PyMethod_New(func, obj, type);
}
static PyObject*
__Pyx_CyFunction_repr(__pyx_CyFunctionObject *op)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("<cyfunction %U at %p>",
                                op->func_qualname, (void *)op);
#else
    return PyString_FromFormat("<cyfunction %s at %p>",
                               PyString_AsString(op->func_qualname), (void *)op);
#endif
}
static PyObject * __Pyx_CyFunction_CallMethod(PyObject *func, PyObject *self, PyObject *arg, PyObject *kw) {
    PyCFunctionObject* f = (PyCFunctionObject*)func;
    PyCFunction meth = f->m_ml->ml_meth;
    Py_ssize_t size;
    switch (f->m_ml->ml_flags & (METH_VARARGS | METH_KEYWORDS | METH_NOARGS | METH_O)) {
    case METH_VARARGS:
        if (likely(kw == NULL || PyDict_Size(kw) == 0))
            return (*meth)(self, arg);
        break;
    case METH_VARARGS | METH_KEYWORDS:
        return (*(PyCFunctionWithKeywords)meth)(self, arg, kw);
    case METH_NOARGS:
        if (likely(kw == NULL || PyDict_Size(kw) == 0)) {
            size = PyTuple_GET_SIZE(arg);
            if (likely(size == 0))
                return (*meth)(self, NULL);
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes no arguments (%" CYTHON_FORMAT_SSIZE_T "d given)",
                f->m_ml->ml_name, size);
            return NULL;
        }
        break;
    case METH_O:
        if (likely(kw == NULL || PyDict_Size(kw) == 0)) {
            size = PyTuple_GET_SIZE(arg);
            if (likely(size == 1)) {
                PyObject *result, *arg0;
                #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
                arg0 = PyTuple_GET_ITEM(arg, 0);
                #else
                arg0 = PySequence_ITEM(arg, 0); if (unlikely(!arg0)) return NULL;
                #endif
                result = (*meth)(self, arg0);
                #if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
                Py_DECREF(arg0);
                #endif
                return result;
            }
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes exactly one argument (%" CYTHON_FORMAT_SSIZE_T "d given)",
                f->m_ml->ml_name, size);
            return NULL;
        }
        break;
    default:
        PyErr_SetString(PyExc_SystemError, "Bad call flags in "
                        "__Pyx_CyFunction_Call. METH_OLDARGS is no "
                        "longer supported!");
        return NULL;
    }
    PyErr_Format(PyExc_TypeError, "%.200s() takes no keyword arguments",
                 f->m_ml->ml_name);
    return NULL;
}
static CYTHON_INLINE PyObject *__Pyx_CyFunction_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    return __Pyx_CyFunction_CallMethod(func, ((PyCFunctionObject*)func)->m_self, arg, kw);
}
static PyObject *__Pyx_CyFunction_CallAsMethod(PyObject *func, PyObject *args, PyObject *kw) {
    PyObject *result;
    __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *) func;
    if ((cyfunc->flags & __Pyx_CYFUNCTION_CCLASS) && !(cyfunc->flags & __Pyx_CYFUNCTION_STATICMETHOD)) {
        Py_ssize_t argc;
        PyObject *new_args;
        PyObject *self;
        argc = PyTuple_GET_SIZE(args);
        new_args = PyTuple_GetSlice(args, 1, argc);
        if (unlikely(!new_args))
            return NULL;
        self = PyTuple_GetItem(args, 0);
        if (unlikely(!self)) {
            Py_DECREF(new_args);
            return NULL;
        }
        result = __Pyx_CyFunction_CallMethod(func, self, new_args, kw);
        Py_DECREF(new_args);
    } else {
        result = __Pyx_CyFunction_Call(func, args, kw);
    }
    return result;
}
static PyTypeObject __pyx_CyFunctionType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "cython_function_or_method",
    sizeof(__pyx_CyFunctionObject),
    0,
    (destructor) __Pyx_CyFunction_dealloc,
    0,
    0,
    0,
#if PY_MAJOR_VERSION < 3
    0,
#else
    0,
#endif
    (reprfunc) __Pyx_CyFunction_repr,
    0,
    0,
    0,
    0,
    __Pyx_CyFunction_CallAsMethod,
    0,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    0,
    (traverseproc) __Pyx_CyFunction_traverse,
    (inquiry) __Pyx_CyFunction_clear,
    0,
#if PY_VERSION_HEX < 0x030500A0
    offsetof(__pyx_CyFunctionObject, func_weakreflist),
#else
    offsetof(PyCFunctionObject, m_weakreflist),
#endif
    0,
    0,
    __pyx_CyFunction_methods,
    __pyx_CyFunction_members,
    __pyx_CyFunction_getsets,
    0,
    0,
    __Pyx_CyFunction_descr_get,
    0,
    offsetof(__pyx_CyFunctionObject, func_dict),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
#if PY_VERSION_HEX >= 0x030400a1
    0,
#endif
};
static int __pyx_CyFunction_init(void) {
    __pyx_CyFunctionType = __Pyx_FetchCommonType(&__pyx_CyFunctionType_type);
    if (unlikely(__pyx_CyFunctionType == NULL)) {
        return -1;
    }
    return 0;
}
static CYTHON_INLINE void *__Pyx_CyFunction_InitDefaults(PyObject *func, size_t size, int pyobjects) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->defaults = PyObject_Malloc(size);
    if (unlikely(!m->defaults))
        return PyErr_NoMemory();
    memset(m->defaults, 0, size);
    m->defaults_pyobjects = pyobjects;
    return m->defaults;
}
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsTuple(PyObject *func, PyObject *tuple) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->defaults_tuple = tuple;
    Py_INCREF(tuple);
}
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsKwDict(PyObject *func, PyObject *dict) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->defaults_kwdict = dict;
    Py_INCREF(dict);
}
static CYTHON_INLINE void __Pyx_CyFunction_SetAnnotationsDict(PyObject *func, PyObject *dict) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->func_annotations = dict;
    Py_INCREF(dict);
}

/* BufferFallbackError */
          static void __Pyx_RaiseBufferFallbackError(void) {
  PyErr_SetString(PyExc_ValueError,
     "Buffer acquisition failed on assignment; and then reacquiring the old buffer failed too!");
}

/* None */
          static CYTHON_INLINE Py_ssize_t __Pyx_div_Py_ssize_t(Py_ssize_t a, Py_ssize_t b) {
    Py_ssize_t q = a / b;
    Py_ssize_t r = a - q*b;
    q -= ((r != 0) & ((r ^ b) < 0));
    return q;
}

/* BufferIndexError */
          static void __Pyx_RaiseBufferIndexError(int axis) {
  PyErr_Format(PyExc_IndexError,
     "Out of bounds on buffer access (axis %d)", axis);
}

/* RaiseTooManyValuesToUnpack */
          static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected) {
    PyErr_Format(PyExc_ValueError,
                 "too many values to unpack (expected %" CYTHON_FORMAT_SSIZE_T "d)", expected);
}

/* RaiseNeedMoreValuesToUnpack */
          static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index) {
    PyErr_Format(PyExc_ValueError,
                 "need more than %" CYTHON_FORMAT_SSIZE_T "d value%.1s to unpack",
                 index, (index == 1) ? "" : "s");
}

/* RaiseNoneIterError */
          static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
}

/* SaveResetException */
          #if CYTHON_FAST_THREAD_STATE
static CYTHON_INLINE void __Pyx__ExceptionSave(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb) {
    #if PY_VERSION_HEX >= 0x030700A3
    *type = tstate->exc_state.exc_type;
    *value = tstate->exc_state.exc_value;
    *tb = tstate->exc_state.exc_traceback;
    #else
    *type = tstate->exc_type;
    *value = tstate->exc_value;
    *tb = tstate->exc_traceback;
    #endif
    Py_XINCREF(*type);
    Py_XINCREF(*value);
    Py_XINCREF(*tb);
}
static CYTHON_INLINE void __Pyx__ExceptionReset(PyThreadState *tstate, PyObject *type, PyObject *value, PyObject *tb) {
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    #if PY_VERSION_HEX >= 0x030700A3
    tmp_type = tstate->exc_state.exc_type;
    tmp_value = tstate->exc_state.exc_value;
    tmp_tb = tstate->exc_state.exc_traceback;
    tstate->exc_state.exc_type = type;
    tstate->exc_state.exc_value = value;
    tstate->exc_state.exc_traceback = tb;
    #else
    tmp_type = tstate->exc_type;
    tmp_value = tstate->exc_value;
    tmp_tb = tstate->exc_traceback;
    tstate->exc_type = type;
    tstate->exc_value = value;
    tstate->exc_traceback = tb;
    #endif
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
}
#endif

/* PyErrExceptionMatches */
          #if CYTHON_FAST_THREAD_STATE
static int __Pyx_PyErr_ExceptionMatchesTuple(PyObject *exc_type, PyObject *tuple) {
    Py_ssize_t i, n;
    n = PyTuple_GET_SIZE(tuple);
#if PY_MAJOR_VERSION >= 3
    for (i=0; i<n; i++) {
        if (exc_type == PyTuple_GET_ITEM(tuple, i)) return 1;
    }
#endif
    for (i=0; i<n; i++) {
        if (__Pyx_PyErr_GivenExceptionMatches(exc_type, PyTuple_GET_ITEM(tuple, i))) return 1;
    }
    return 0;
}
static CYTHON_INLINE int __Pyx_PyErr_ExceptionMatchesInState(PyThreadState* tstate, PyObject* err) {
    PyObject *exc_type = tstate->curexc_type;
    if (exc_type == err) return 1;
    if (unlikely(!exc_type)) return 0;
    if (unlikely(PyTuple_Check(err)))
        return __Pyx_PyErr_ExceptionMatchesTuple(exc_type, err);
    return __Pyx_PyErr_GivenExceptionMatches(exc_type, err);
}
#endif

/* GetException */
          #if CYTHON_FAST_THREAD_STATE
static int __Pyx__GetException(PyThreadState *tstate, PyObject **type, PyObject **value, PyObject **tb) {
#else
static int __Pyx_GetException(PyObject **type, PyObject **value, PyObject **tb) {
#endif
    PyObject *local_type, *local_value, *local_tb;
#if CYTHON_FAST_THREAD_STATE
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    local_type = tstate->curexc_type;
    local_value = tstate->curexc_value;
    local_tb = tstate->curexc_traceback;
    tstate->curexc_type = 0;
    tstate->curexc_value = 0;
    tstate->curexc_traceback = 0;
#else
    PyErr_Fetch(&local_type, &local_value, &local_tb);
#endif
    PyErr_NormalizeException(&local_type, &local_value, &local_tb);
#if CYTHON_FAST_THREAD_STATE
    if (unlikely(tstate->curexc_type))
#else
    if (unlikely(PyErr_Occurred()))
#endif
        goto bad;
    #if PY_MAJOR_VERSION >= 3
    if (local_tb) {
        if (unlikely(PyException_SetTraceback(local_value, local_tb) < 0))
            goto bad;
    }
    #endif
    Py_XINCREF(local_tb);
    Py_XINCREF(local_type);
    Py_XINCREF(local_value);
    *type = local_type;
    *value = local_value;
    *tb = local_tb;
#if CYTHON_FAST_THREAD_STATE
    #if PY_VERSION_HEX >= 0x030700A3
    tmp_type = tstate->exc_state.exc_type;
    tmp_value = tstate->exc_state.exc_value;
    tmp_tb = tstate->exc_state.exc_traceback;
    tstate->exc_state.exc_type = local_type;
    tstate->exc_state.exc_value = local_value;
    tstate->exc_state.exc_traceback = local_tb;
    #else
    tmp_type = tstate->exc_type;
    tmp_value = tstate->exc_value;
    tmp_tb = tstate->exc_traceback;
    tstate->exc_type = local_type;
    tstate->exc_value = local_value;
    tstate->exc_traceback = local_tb;
    #endif
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
#else
    PyErr_SetExcInfo(local_type, local_value, local_tb);
#endif
    return 0;
bad:
    *type = 0;
    *value = 0;
    *tb = 0;
    Py_XDECREF(local_type);
    Py_XDECREF(local_value);
    Py_XDECREF(local_tb);
    return -1;
}

/* PyObject_GenericGetAttrNoDict */
            #if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000
static PyObject *__Pyx_RaiseGenericGetAttributeError(PyTypeObject *tp, PyObject *attr_name) {
    PyErr_Format(PyExc_AttributeError,
#if PY_MAJOR_VERSION >= 3
                 "'%.50s' object has no attribute '%U'",
                 tp->tp_name, attr_name);
#else
                 "'%.50s' object has no attribute '%.400s'",
                 tp->tp_name, PyString_AS_STRING(attr_name));
#endif
    return NULL;
}
static CYTHON_INLINE PyObject* __Pyx_PyObject_GenericGetAttrNoDict(PyObject* obj, PyObject* attr_name) {
    PyObject *descr;
    PyTypeObject *tp = Py_TYPE(obj);
    if (unlikely(!PyString_Check(attr_name))) {
        return PyObject_GenericGetAttr(obj, attr_name);
    }
    assert(!tp->tp_dictoffset);
    descr = _PyType_Lookup(tp, attr_name);
    if (unlikely(!descr)) {
        return __Pyx_RaiseGenericGetAttributeError(tp, attr_name);
    }
    Py_INCREF(descr);
    #if PY_MAJOR_VERSION < 3
    if (likely(PyType_HasFeature(Py_TYPE(descr), Py_TPFLAGS_HAVE_CLASS)))
    #endif
    {
        descrgetfunc f = Py_TYPE(descr)->tp_descr_get;
        if (unlikely(f)) {
            PyObject *res = f(descr, obj, (PyObject *)tp);
            Py_DECREF(descr);
            return res;
        }
    }
    return descr;
}
#endif

/* PyObject_GenericGetAttr */
            #if CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP && PY_VERSION_HEX < 0x03070000
static PyObject* __Pyx_PyObject_GenericGetAttr(PyObject* obj, PyObject* attr_name) {
    if (unlikely(Py_TYPE(obj)->tp_dictoffset)) {
        return PyObject_GenericGetAttr(obj, attr_name);
    }
    return __Pyx_PyObject_GenericGetAttrNoDict(obj, attr_name);
}
#endif

/* SetupReduce */
            static int __Pyx_setup_reduce_is_named(PyObject* meth, PyObject* name) {
  int ret;
  PyObject *name_attr;
  name_attr = __Pyx_PyObject_GetAttrStr(meth, __pyx_n_s_name);
  if (likely(name_attr)) {
      ret = PyObject_RichCompareBool(name_attr, name, Py_EQ);
  } else {
      ret = -1;
  }
  if (unlikely(ret < 0)) {
      PyErr_Clear();
      ret = 0;
  }
  Py_XDECREF(name_attr);
  return ret;
}
static int __Pyx_setup_reduce(PyObject* type_obj) {
    int ret = 0;
    PyObject *object_reduce = NULL;
    PyObject *object_reduce_ex = NULL;
    PyObject *reduce = NULL;
    PyObject *reduce_ex = NULL;
    PyObject *reduce_cython = NULL;
    PyObject *setstate = NULL;
    PyObject *setstate_cython = NULL;
#if CYTHON_USE_PYTYPE_LOOKUP
    if (_PyType_Lookup((PyTypeObject*)type_obj, __pyx_n_s_getstate)) goto GOOD;
#else
    if (PyObject_HasAttr(type_obj, __pyx_n_s_getstate)) goto GOOD;
#endif
#if CYTHON_USE_PYTYPE_LOOKUP
    object_reduce_ex = _PyType_Lookup(&PyBaseObject_Type, __pyx_n_s_reduce_ex); if (!object_reduce_ex) goto BAD;
#else
    object_reduce_ex = __Pyx_PyObject_GetAttrStr((PyObject*)&PyBaseObject_Type, __pyx_n_s_reduce_ex); if (!object_reduce_ex) goto BAD;
#endif
    reduce_ex = __Pyx_PyObject_GetAttrStr(type_obj, __pyx_n_s_reduce_ex); if (unlikely(!reduce_ex)) goto BAD;
    if (reduce_ex == object_reduce_ex) {
#if CYTHON_USE_PYTYPE_LOOKUP
        object_reduce = _PyType_Lookup(&PyBaseObject_Type, __pyx_n_s_reduce); if (!object_reduce) goto BAD;
#else
        object_reduce = __Pyx_PyObject_GetAttrStr((PyObject*)&PyBaseObject_Type, __pyx_n_s_reduce); if (!object_reduce) goto BAD;
#endif
        reduce = __Pyx_PyObject_GetAttrStr(type_obj, __pyx_n_s_reduce); if (unlikely(!reduce)) goto BAD;
        if (reduce == object_reduce || __Pyx_setup_reduce_is_named(reduce, __pyx_n_s_reduce_cython)) {
            reduce_cython = __Pyx_PyObject_GetAttrStr(type_obj, __pyx_n_s_reduce_cython); if (unlikely(!reduce_cython)) goto BAD;
            ret = PyDict_SetItem(((PyTypeObject*)type_obj)->tp_dict, __pyx_n_s_reduce, reduce_cython); if (unlikely(ret < 0)) goto BAD;
            ret = PyDict_DelItem(((PyTypeObject*)type_obj)->tp_dict, __pyx_n_s_reduce_cython); if (unlikely(ret < 0)) goto BAD;
            setstate = __Pyx_PyObject_GetAttrStr(type_obj, __pyx_n_s_setstate);
            if (!setstate) PyErr_Clear();
            if (!setstate || __Pyx_setup_reduce_is_named(setstate, __pyx_n_s_setstate_cython)) {
                setstate_cython = __Pyx_PyObject_GetAttrStr(type_obj, __pyx_n_s_setstate_cython); if (unlikely(!setstate_cython)) goto BAD;
                ret = PyDict_SetItem(((PyTypeObject*)type_obj)->tp_dict, __pyx_n_s_setstate, setstate_cython); if (unlikely(ret < 0)) goto BAD;
                ret = PyDict_DelItem(((PyTypeObject*)type_obj)->tp_dict, __pyx_n_s_setstate_cython); if (unlikely(ret < 0)) goto BAD;
            }
            PyType_Modified((PyTypeObject*)type_obj);
        }
    }
    goto GOOD;
BAD:
    if (!PyErr_Occurred())
        PyErr_Format(PyExc_RuntimeError, "Unable to initialize pickling for %s", ((PyTypeObject*)type_obj)->tp_name);
    ret = -1;
GOOD:
#if !CYTHON_USE_PYTYPE_LOOKUP
    Py_XDECREF(object_reduce);
    Py_XDECREF(object_reduce_ex);
#endif
    Py_XDECREF(reduce);
    Py_XDECREF(reduce_ex);
    Py_XDECREF(reduce_cython);
    Py_XDECREF(setstate);
    Py_XDECREF(setstate_cython);
    return ret;
}

/* Import */
            static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level) {
    PyObject *empty_list = 0;
    PyObject *module = 0;
    PyObject *global_dict = 0;
    PyObject *empty_dict = 0;
    PyObject *list;
    #if PY_MAJOR_VERSION < 3
    PyObject *py_import;
    py_import = __Pyx_PyObject_GetAttrStr(__pyx_b, __pyx_n_s_import);
    if (!py_import)
        goto bad;
    #endif
    if (from_list)
        list = from_list;
    else {
        empty_list = PyList_New(0);
        if (!empty_list)
            goto bad;
        list = empty_list;
    }
    global_dict = PyModule_GetDict(__pyx_m);
    if (!global_dict)
        goto bad;
    empty_dict = PyDict_New();
    if (!empty_dict)
        goto bad;
    {
        #if PY_MAJOR_VERSION >= 3
        if (level == -1) {
            if (strchr(__Pyx_MODULE_NAME, '.')) {
                module = PyImport_ImportModuleLevelObject(
                    name, global_dict, empty_dict, list, 1);
                if (!module) {
                    if (!PyErr_ExceptionMatches(PyExc_ImportError))
                        goto bad;
                    PyErr_Clear();
                }
            }
            level = 0;
        }
        #endif
        if (!module) {
            #if PY_MAJOR_VERSION < 3
            PyObject *py_level = PyInt_FromLong(level);
            if (!py_level)
                goto bad;
            module = PyObject_CallFunctionObjArgs(py_import,
                name, global_dict, empty_dict, list, py_level, NULL);
            Py_DECREF(py_level);
            #else
            module = PyImport_ImportModuleLevelObject(
                name, global_dict, empty_dict, list, level);
            #endif
        }
    }
bad:
    #if PY_MAJOR_VERSION < 3
    Py_XDECREF(py_import);
    #endif
    Py_XDECREF(empty_list);
    Py_XDECREF(empty_dict);
    return module;
}

/* CLineInTraceback */
            #ifndef CYTHON_CLINE_IN_TRACEBACK
static int __Pyx_CLineForTraceback(CYTHON_UNUSED PyThreadState *tstate, int c_line) {
    PyObject *use_cline;
    PyObject *ptype, *pvalue, *ptraceback;
#if CYTHON_COMPILING_IN_CPYTHON
    PyObject **cython_runtime_dict;
#endif
    if (unlikely(!__pyx_cython_runtime)) {
        return c_line;
    }
    __Pyx_ErrFetchInState(tstate, &ptype, &pvalue, &ptraceback);
#if CYTHON_COMPILING_IN_CPYTHON
    cython_runtime_dict = _PyObject_GetDictPtr(__pyx_cython_runtime);
    if (likely(cython_runtime_dict)) {
      use_cline = __Pyx_PyDict_GetItemStr(*cython_runtime_dict, __pyx_n_s_cline_in_traceback);
    } else
#endif
    {
      PyObject *use_cline_obj = __Pyx_PyObject_GetAttrStr(__pyx_cython_runtime, __pyx_n_s_cline_in_traceback);
      if (use_cline_obj) {
        use_cline = PyObject_Not(use_cline_obj) ? Py_False : Py_True;
        Py_DECREF(use_cline_obj);
      } else {
        PyErr_Clear();
        use_cline = NULL;
      }
    }
    if (!use_cline) {
        c_line = 0;
        PyObject_SetAttr(__pyx_cython_runtime, __pyx_n_s_cline_in_traceback, Py_False);
    }
    else if (PyObject_Not(use_cline) != 0) {
        c_line = 0;
    }
    __Pyx_ErrRestoreInState(tstate, ptype, pvalue, ptraceback);
    return c_line;
}
#endif

/* CodeObjectCache */
            static int __pyx_bisect_code_objects(__Pyx_CodeObjectCacheEntry* entries, int count, int code_line) {
    int start = 0, mid = 0, end = count - 1;
    if (end >= 0 && code_line > entries[end].code_line) {
        return count;
    }
    while (start < end) {
        mid = start + (end - start) / 2;
        if (code_line < entries[mid].code_line) {
            end = mid;
        } else if (code_line > entries[mid].code_line) {
             start = mid + 1;
        } else {
            return mid;
        }
    }
    if (code_line <= entries[mid].code_line) {
        return mid;
    } else {
        return mid + 1;
    }
}
static PyCodeObject *__pyx_find_code_object(int code_line) {
    PyCodeObject* code_object;
    int pos;
    if (unlikely(!code_line) || unlikely(!__pyx_code_cache.entries)) {
        return NULL;
    }
    pos = __pyx_bisect_code_objects(__pyx_code_cache.entries, __pyx_code_cache.count, code_line);
    if (unlikely(pos >= __pyx_code_cache.count) || unlikely(__pyx_code_cache.entries[pos].code_line != code_line)) {
        return NULL;
    }
    code_object = __pyx_code_cache.entries[pos].code_object;
    Py_INCREF(code_object);
    return code_object;
}
static void __pyx_insert_code_object(int code_line, PyCodeObject* code_object) {
    int pos, i;
    __Pyx_CodeObjectCacheEntry* entries = __pyx_code_cache.entries;
    if (unlikely(!code_line)) {
        return;
    }
    if (unlikely(!entries)) {
        entries = (__Pyx_CodeObjectCacheEntry*)PyMem_Malloc(64*sizeof(__Pyx_CodeObjectCacheEntry));
        if (likely(entries)) {
            __pyx_code_cache.entries = entries;
            __pyx_code_cache.max_count = 64;
            __pyx_code_cache.count = 1;
            entries[0].code_line = code_line;
            entries[0].code_object = code_object;
            Py_INCREF(code_object);
        }
        return;
    }
    pos = __pyx_bisect_code_objects(__pyx_code_cache.entries, __pyx_code_cache.count, code_line);
    if ((pos < __pyx_code_cache.count) && unlikely(__pyx_code_cache.entries[pos].code_line == code_line)) {
        PyCodeObject* tmp = entries[pos].code_object;
        entries[pos].code_object = code_object;
        Py_DECREF(tmp);
        return;
    }
    if (__pyx_code_cache.count == __pyx_code_cache.max_count) {
        int new_max = __pyx_code_cache.max_count + 64;
        entries = (__Pyx_CodeObjectCacheEntry*)PyMem_Realloc(
            __pyx_code_cache.entries, (size_t)new_max*sizeof(__Pyx_CodeObjectCacheEntry));
        if (unlikely(!entries)) {
            return;
        }
        __pyx_code_cache.entries = entries;
        __pyx_code_cache.max_count = new_max;
    }
    for (i=__pyx_code_cache.count; i>pos; i--) {
        entries[i] = entries[i-1];
    }
    entries[pos].code_line = code_line;
    entries[pos].code_object = code_object;
    __pyx_code_cache.count++;
    Py_INCREF(code_object);
}

/* AddTraceback */
            #include "compile.h"
#include "frameobject.h"
#include "traceback.h"
static PyCodeObject* __Pyx_CreateCodeObjectForTraceback(
            const char *funcname, int c_line,
            int py_line, const char *filename) {
    PyCodeObject *py_code = 0;
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;
    #if PY_MAJOR_VERSION < 3
    py_srcfile = PyString_FromString(filename);
    #else
    py_srcfile = PyUnicode_FromString(filename);
    #endif
    if (!py_srcfile) goto bad;
    if (c_line) {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, c_line);
        #else
        py_funcname = PyUnicode_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, c_line);
        #endif
    }
    else {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromString(funcname);
        #else
        py_funcname = PyUnicode_FromString(funcname);
        #endif
    }
    if (!py_funcname) goto bad;
    py_code = __Pyx_PyCode_New(
        0,
        0,
        0,
        0,
        0,
        __pyx_empty_bytes, /*PyObject *code,*/
        __pyx_empty_tuple, /*PyObject *consts,*/
        __pyx_empty_tuple, /*PyObject *names,*/
        __pyx_empty_tuple, /*PyObject *varnames,*/
        __pyx_empty_tuple, /*PyObject *freevars,*/
        __pyx_empty_tuple, /*PyObject *cellvars,*/
        py_srcfile,   /*PyObject *filename,*/
        py_funcname,  /*PyObject *name,*/
        py_line,
        __pyx_empty_bytes  /*PyObject *lnotab*/
    );
    Py_DECREF(py_srcfile);
    Py_DECREF(py_funcname);
    return py_code;
bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
    return NULL;
}
static void __Pyx_AddTraceback(const char *funcname, int c_line,
                               int py_line, const char *filename) {
    PyCodeObject *py_code = 0;
    PyFrameObject *py_frame = 0;
    PyThreadState *tstate = __Pyx_PyThreadState_Current;
    if (c_line) {
        c_line = __Pyx_CLineForTraceback(tstate, c_line);
    }
    py_code = __pyx_find_code_object(c_line ? -c_line : py_line);
    if (!py_code) {
        py_code = __Pyx_CreateCodeObjectForTraceback(
            funcname, c_line, py_line, filename);
        if (!py_code) goto bad;
        __pyx_insert_code_object(c_line ? -c_line : py_line, py_code);
    }
    py_frame = PyFrame_New(
        tstate,            /*PyThreadState *tstate,*/
        py_code,           /*PyCodeObject *code,*/
        __pyx_d,    /*PyObject *globals,*/
        0                  /*PyObject *locals*/
    );
    if (!py_frame) goto bad;
    __Pyx_PyFrame_SetLineNumber(py_frame, py_line);
    PyTraceBack_Here(py_frame);
bad:
    Py_XDECREF(py_code);
    Py_XDECREF(py_frame);
}

#if PY_MAJOR_VERSION < 3
static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags) {
    if (PyObject_CheckBuffer(obj)) return PyObject_GetBuffer(obj, view, flags);
        if (__Pyx_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) return __pyx_pw_5numpy_7ndarray_1__getbuffer__(obj, view, flags);
    PyErr_Format(PyExc_TypeError, "'%.200s' does not have the buffer interface", Py_TYPE(obj)->tp_name);
    return -1;
}
static void __Pyx_ReleaseBuffer(Py_buffer *view) {
    PyObject *obj = view->obj;
    if (!obj) return;
    if (PyObject_CheckBuffer(obj)) {
        PyBuffer_Release(view);
        return;
    }
    if ((0)) {}
        else if (__Pyx_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) __pyx_pw_5numpy_7ndarray_3__releasebuffer__(obj, view);
    view->obj = NULL;
    Py_DECREF(obj);
}
#endif


            /* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_long(long value) {
    const long neg_one = (long) -1, const_zero = (long) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(long) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(long) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(long) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(long) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(long) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(long),
                                     little, !is_unsigned);
    }
}

/* CIntFromPyVerify */
            #define __PYX_VERIFY_RETURN_INT(target_type, func_type, func_value)\
    __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, 0)
#define __PYX_VERIFY_RETURN_INT_EXC(target_type, func_type, func_value)\
    __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, 1)
#define __PYX__VERIFY_RETURN_INT(target_type, func_type, func_value, exc)\
    {\
        func_type value = func_value;\
        if (sizeof(target_type) < sizeof(func_type)) {\
            if (unlikely(value != (func_type) (target_type) value)) {\
                func_type zero = 0;\
                if (exc && unlikely(value == (func_type)-1 && PyErr_Occurred()))\
                    return (target_type) -1;\
                if (is_unsigned && unlikely(value < zero))\
                    goto raise_neg_overflow;\
                else\
                    goto raise_overflow;\
            }\
        }\
        return (target_type) value;\
    }

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_siz(siz value) {
    const siz neg_one = (siz) -1, const_zero = (siz) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(siz) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(siz) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(siz) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(siz) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(siz) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(siz),
                                     little, !is_unsigned);
    }
}

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_Py_intptr_t(Py_intptr_t value) {
    const Py_intptr_t neg_one = (Py_intptr_t) -1, const_zero = (Py_intptr_t) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(Py_intptr_t) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(Py_intptr_t) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(Py_intptr_t) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(Py_intptr_t) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(Py_intptr_t) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(Py_intptr_t),
                                     little, !is_unsigned);
    }
}

/* Declarations */
            #if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return ::std::complex< float >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return x + y*(__pyx_t_float_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      __pyx_t_float_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

/* Arithmetic */
            #if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sum_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_diff_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prod_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    #if 1
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quot_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        if (b.imag == 0) {
            return __pyx_t_float_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else if (fabsf(b.real) >= fabsf(b.imag)) {
            if (b.real == 0 && b.imag == 0) {
                return __pyx_t_float_complex_from_parts(a.real / b.real, a.imag / b.imag);
            } else {
                float r = b.imag / b.real;
                float s = 1.0 / (b.real + b.imag * r);
                return __pyx_t_float_complex_from_parts(
                    (a.real + a.imag * r) * s, (a.imag - a.real * r) * s);
            }
        } else {
            float r = b.real / b.imag;
            float s = 1.0 / (b.imag + b.real * r);
            return __pyx_t_float_complex_from_parts(
                (a.real * r + a.imag) * s, (a.imag * r - a.real) * s);
        }
    }
    #else
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quot_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        if (b.imag == 0) {
            return __pyx_t_float_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else {
            float denom = b.real * b.real + b.imag * b.imag;
            return __pyx_t_float_complex_from_parts(
                (a.real * b.real + a.imag * b.imag) / denom,
                (a.imag * b.real - a.real * b.imag) / denom);
        }
    }
    #endif
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_neg_float(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero_float(__pyx_t_float_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conj_float(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE float __Pyx_c_abs_float(__pyx_t_float_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrtf(z.real*z.real + z.imag*z.imag);
          #else
            return hypotf(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_pow_float(__pyx_t_float_complex a, __pyx_t_float_complex b) {
            __pyx_t_float_complex z;
            float r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    float denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prod_float(a, a);
                        return __Pyx_c_prod_float(a, a);
                    case 3:
                        z = __Pyx_c_prod_float(a, a);
                        return __Pyx_c_prod_float(z, a);
                    case 4:
                        z = __Pyx_c_prod_float(a, a);
                        return __Pyx_c_prod_float(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                } else if (b.imag == 0) {
                    z.real = powf(a.real, b.real);
                    z.imag = 0;
                    return z;
                } else if (a.real > 0) {
                    r = a.real;
                    theta = 0;
                } else {
                    r = -a.real;
                    theta = atan2f(0, -1);
                }
            } else {
                r = __Pyx_c_abs_float(a);
                theta = atan2f(a.imag, a.real);
            }
            lnr = logf(r);
            z_r = expf(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cosf(z_theta);
            z.imag = z_r * sinf(z_theta);
            return z;
        }
    #endif
#endif

/* Declarations */
            #if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return ::std::complex< double >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return x + y*(__pyx_t_double_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      __pyx_t_double_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

/* Arithmetic */
            #if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    #if 1
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        if (b.imag == 0) {
            return __pyx_t_double_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else if (fabs(b.real) >= fabs(b.imag)) {
            if (b.real == 0 && b.imag == 0) {
                return __pyx_t_double_complex_from_parts(a.real / b.real, a.imag / b.imag);
            } else {
                double r = b.imag / b.real;
                double s = 1.0 / (b.real + b.imag * r);
                return __pyx_t_double_complex_from_parts(
                    (a.real + a.imag * r) * s, (a.imag - a.real * r) * s);
            }
        } else {
            double r = b.real / b.imag;
            double s = 1.0 / (b.imag + b.real * r);
            return __pyx_t_double_complex_from_parts(
                (a.real * r + a.imag) * s, (a.imag * r - a.real) * s);
        }
    }
    #else
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        if (b.imag == 0) {
            return __pyx_t_double_complex_from_parts(a.real / b.real, a.imag / b.real);
        } else {
            double denom = b.real * b.real + b.imag * b.imag;
            return __pyx_t_double_complex_from_parts(
                (a.real * b.real + a.imag * b.imag) / denom,
                (a.imag * b.real - a.real * b.imag) / denom);
        }
    }
    #endif
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg_double(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero_double(__pyx_t_double_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj_double(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs_double(__pyx_t_double_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrt(z.real*z.real + z.imag*z.imag);
          #else
            return hypot(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow_double(__pyx_t_double_complex a, __pyx_t_double_complex b) {
            __pyx_t_double_complex z;
            double r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    double denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prod_double(a, a);
                        return __Pyx_c_prod_double(a, a);
                    case 3:
                        z = __Pyx_c_prod_double(a, a);
                        return __Pyx_c_prod_double(z, a);
                    case 4:
                        z = __Pyx_c_prod_double(a, a);
                        return __Pyx_c_prod_double(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                } else if (b.imag == 0) {
                    z.real = pow(a.real, b.real);
                    z.imag = 0;
                    return z;
                } else if (a.real > 0) {
                    r = a.real;
                    theta = 0;
                } else {
                    r = -a.real;
                    theta = atan2(0, -1);
                }
            } else {
                r = __Pyx_c_abs_double(a);
                theta = atan2(a.imag, a.real);
            }
            lnr = log(r);
            z_r = exp(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cos(z_theta);
            z.imag = z_r * sin(z_theta);
            return z;
        }
    #endif
#endif

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_int(int value) {
    const int neg_one = (int) -1, const_zero = (int) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(int) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(int) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(int) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(int) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(int) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(int),
                                     little, !is_unsigned);
    }
}

/* CIntToPy */
            static CYTHON_INLINE PyObject* __Pyx_PyInt_From_enum__NPY_TYPES(enum NPY_TYPES value) {
    const enum NPY_TYPES neg_one = (enum NPY_TYPES) -1, const_zero = (enum NPY_TYPES) 0;
    const int is_unsigned = neg_one > const_zero;
    if (is_unsigned) {
        if (sizeof(enum NPY_TYPES) < sizeof(long)) {
            return PyInt_FromLong((long) value);
        } else if (sizeof(enum NPY_TYPES) <= sizeof(unsigned long)) {
            return PyLong_FromUnsignedLong((unsigned long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(enum NPY_TYPES) <= sizeof(unsigned PY_LONG_LONG)) {
            return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG) value);
#endif
        }
    } else {
        if (sizeof(enum NPY_TYPES) <= sizeof(long)) {
            return PyInt_FromLong((long) value);
#ifdef HAVE_LONG_LONG
        } else if (sizeof(enum NPY_TYPES) <= sizeof(PY_LONG_LONG)) {
            return PyLong_FromLongLong((PY_LONG_LONG) value);
#endif
        }
    }
    {
        int one = 1; int little = (int)*(unsigned char *)&one;
        unsigned char *bytes = (unsigned char *)&value;
        return _PyLong_FromByteArray(bytes, sizeof(enum NPY_TYPES),
                                     little, !is_unsigned);
    }
}

/* CIntFromPy */
            static CYTHON_INLINE siz __Pyx_PyInt_As_siz(PyObject *x) {
    const siz neg_one = (siz) -1, const_zero = (siz) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(siz) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(siz, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (siz) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (siz) 0;
                case  1: __PYX_VERIFY_RETURN_INT(siz, digit, digits[0])
                case 2:
                    if (8 * sizeof(siz) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) >= 2 * PyLong_SHIFT) {
                            return (siz) (((((siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(siz) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) >= 3 * PyLong_SHIFT) {
                            return (siz) (((((((siz)digits[2]) << PyLong_SHIFT) | (siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(siz) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) >= 4 * PyLong_SHIFT) {
                            return (siz) (((((((((siz)digits[3]) << PyLong_SHIFT) | (siz)digits[2]) << PyLong_SHIFT) | (siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (siz) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(siz) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(siz, unsigned long, PyLong_AsUnsignedLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(siz) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(siz, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
#endif
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (siz) 0;
                case -1: __PYX_VERIFY_RETURN_INT(siz, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(siz,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(siz) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) - 1 > 2 * PyLong_SHIFT) {
                            return (siz) (((siz)-1)*(((((siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(siz) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) - 1 > 2 * PyLong_SHIFT) {
                            return (siz) ((((((siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(siz) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) - 1 > 3 * PyLong_SHIFT) {
                            return (siz) (((siz)-1)*(((((((siz)digits[2]) << PyLong_SHIFT) | (siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(siz) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) - 1 > 3 * PyLong_SHIFT) {
                            return (siz) ((((((((siz)digits[2]) << PyLong_SHIFT) | (siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(siz) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) - 1 > 4 * PyLong_SHIFT) {
                            return (siz) (((siz)-1)*(((((((((siz)digits[3]) << PyLong_SHIFT) | (siz)digits[2]) << PyLong_SHIFT) | (siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(siz) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(siz, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(siz) - 1 > 4 * PyLong_SHIFT) {
                            return (siz) ((((((((((siz)digits[3]) << PyLong_SHIFT) | (siz)digits[2]) << PyLong_SHIFT) | (siz)digits[1]) << PyLong_SHIFT) | (siz)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(siz) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(siz, long, PyLong_AsLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(siz) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(siz, PY_LONG_LONG, PyLong_AsLongLong(x))
#endif
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            siz val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (siz) -1;
        }
    } else {
        siz val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (siz) -1;
        val = __Pyx_PyInt_As_siz(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to siz");
    return (siz) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to siz");
    return (siz) -1;
}

/* CIntFromPy */
            static CYTHON_INLINE size_t __Pyx_PyInt_As_size_t(PyObject *x) {
    const size_t neg_one = (size_t) -1, const_zero = (size_t) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(size_t) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(size_t, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (size_t) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (size_t) 0;
                case  1: __PYX_VERIFY_RETURN_INT(size_t, digit, digits[0])
                case 2:
                    if (8 * sizeof(size_t) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) >= 2 * PyLong_SHIFT) {
                            return (size_t) (((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(size_t) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) >= 3 * PyLong_SHIFT) {
                            return (size_t) (((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(size_t) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) >= 4 * PyLong_SHIFT) {
                            return (size_t) (((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (size_t) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(size_t) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(size_t, unsigned long, PyLong_AsUnsignedLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(size_t) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(size_t, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
#endif
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (size_t) 0;
                case -1: __PYX_VERIFY_RETURN_INT(size_t, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(size_t,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(size_t) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) - 1 > 2 * PyLong_SHIFT) {
                            return (size_t) (((size_t)-1)*(((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(size_t) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) - 1 > 2 * PyLong_SHIFT) {
                            return (size_t) ((((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(size_t) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) - 1 > 3 * PyLong_SHIFT) {
                            return (size_t) (((size_t)-1)*(((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(size_t) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) - 1 > 3 * PyLong_SHIFT) {
                            return (size_t) ((((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(size_t) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) - 1 > 4 * PyLong_SHIFT) {
                            return (size_t) (((size_t)-1)*(((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(size_t) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(size_t, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(size_t) - 1 > 4 * PyLong_SHIFT) {
                            return (size_t) ((((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(size_t) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(size_t, long, PyLong_AsLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(size_t) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(size_t, PY_LONG_LONG, PyLong_AsLongLong(x))
#endif
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            size_t val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (size_t) -1;
        }
    } else {
        size_t val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (size_t) -1;
        val = __Pyx_PyInt_As_size_t(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to size_t");
    return (size_t) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to size_t");
    return (size_t) -1;
}

/* CIntFromPy */
            static CYTHON_INLINE int __Pyx_PyInt_As_int(PyObject *x) {
    const int neg_one = (int) -1, const_zero = (int) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(int) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(int, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (int) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (int) 0;
                case  1: __PYX_VERIFY_RETURN_INT(int, digit, digits[0])
                case 2:
                    if (8 * sizeof(int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 2 * PyLong_SHIFT) {
                            return (int) (((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 3 * PyLong_SHIFT) {
                            return (int) (((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) >= 4 * PyLong_SHIFT) {
                            return (int) (((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (int) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(int) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, unsigned long, PyLong_AsUnsignedLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(int) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
#endif
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (int) 0;
                case -1: __PYX_VERIFY_RETURN_INT(int, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(int,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(int) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(int) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                            return (int) ((((((int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(int) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(int) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                            return (int) ((((((((int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(int) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 4 * PyLong_SHIFT) {
                            return (int) (((int)-1)*(((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(int) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(int, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(int) - 1 > 4 * PyLong_SHIFT) {
                            return (int) ((((((((((int)digits[3]) << PyLong_SHIFT) | (int)digits[2]) << PyLong_SHIFT) | (int)digits[1]) << PyLong_SHIFT) | (int)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(int) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, long, PyLong_AsLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(int) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(int, PY_LONG_LONG, PyLong_AsLongLong(x))
#endif
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            int val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (int) -1;
        }
    } else {
        int val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (int) -1;
        val = __Pyx_PyInt_As_int(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to int");
    return (int) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to int");
    return (int) -1;
}

/* CIntFromPy */
            static CYTHON_INLINE long __Pyx_PyInt_As_long(PyObject *x) {
    const long neg_one = (long) -1, const_zero = (long) 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_Check(x))) {
        if (sizeof(long) < sizeof(long)) {
            __PYX_VERIFY_RETURN_INT(long, long, PyInt_AS_LONG(x))
        } else {
            long val = PyInt_AS_LONG(x);
            if (is_unsigned && unlikely(val < 0)) {
                goto raise_neg_overflow;
            }
            return (long) val;
        }
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (long) 0;
                case  1: __PYX_VERIFY_RETURN_INT(long, digit, digits[0])
                case 2:
                    if (8 * sizeof(long) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 2 * PyLong_SHIFT) {
                            return (long) (((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(long) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 3 * PyLong_SHIFT) {
                            return (long) (((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(long) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) >= 4 * PyLong_SHIFT) {
                            return (long) (((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0]));
                        }
                    }
                    break;
            }
#endif
#if CYTHON_COMPILING_IN_CPYTHON
            if (unlikely(Py_SIZE(x) < 0)) {
                goto raise_neg_overflow;
            }
#else
            {
                int result = PyObject_RichCompareBool(x, Py_False, Py_LT);
                if (unlikely(result < 0))
                    return (long) -1;
                if (unlikely(result == 1))
                    goto raise_neg_overflow;
            }
#endif
            if (sizeof(long) <= sizeof(unsigned long)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, unsigned long, PyLong_AsUnsignedLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(long) <= sizeof(unsigned PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong(x))
#endif
            }
        } else {
#if CYTHON_USE_PYLONG_INTERNALS
            const digit* digits = ((PyLongObject*)x)->ob_digit;
            switch (Py_SIZE(x)) {
                case  0: return (long) 0;
                case -1: __PYX_VERIFY_RETURN_INT(long, sdigit, (sdigit) (-(sdigit)digits[0]))
                case  1: __PYX_VERIFY_RETURN_INT(long,  digit, +digits[0])
                case -2:
                    if (8 * sizeof(long) - 1 > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 2:
                    if (8 * sizeof(long) > 1 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 2 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                            return (long) ((((((long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case -3:
                    if (8 * sizeof(long) - 1 > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 3:
                    if (8 * sizeof(long) > 2 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 3 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                            return (long) ((((((((long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case -4:
                    if (8 * sizeof(long) - 1 > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, long, -(long) (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                            return (long) (((long)-1)*(((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
                case 4:
                    if (8 * sizeof(long) > 3 * PyLong_SHIFT) {
                        if (8 * sizeof(unsigned long) > 4 * PyLong_SHIFT) {
                            __PYX_VERIFY_RETURN_INT(long, unsigned long, (((((((((unsigned long)digits[3]) << PyLong_SHIFT) | (unsigned long)digits[2]) << PyLong_SHIFT) | (unsigned long)digits[1]) << PyLong_SHIFT) | (unsigned long)digits[0])))
                        } else if (8 * sizeof(long) - 1 > 4 * PyLong_SHIFT) {
                            return (long) ((((((((((long)digits[3]) << PyLong_SHIFT) | (long)digits[2]) << PyLong_SHIFT) | (long)digits[1]) << PyLong_SHIFT) | (long)digits[0])));
                        }
                    }
                    break;
            }
#endif
            if (sizeof(long) <= sizeof(long)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, long, PyLong_AsLong(x))
#ifdef HAVE_LONG_LONG
            } else if (sizeof(long) <= sizeof(PY_LONG_LONG)) {
                __PYX_VERIFY_RETURN_INT_EXC(long, PY_LONG_LONG, PyLong_AsLongLong(x))
#endif
            }
        }
        {
#if CYTHON_COMPILING_IN_PYPY && !defined(_PyLong_AsByteArray)
            PyErr_SetString(PyExc_RuntimeError,
                            "_PyLong_AsByteArray() not available in PyPy, cannot convert large numbers");
#else
            long val;
            PyObject *v = __Pyx_PyNumber_IntOrLong(x);
 #if PY_MAJOR_VERSION < 3
            if (likely(v) && !PyLong_Check(v)) {
                PyObject *tmp = v;
                v = PyNumber_Long(tmp);
                Py_DECREF(tmp);
            }
 #endif
            if (likely(v)) {
                int one = 1; int is_little = (int)*(unsigned char *)&one;
                unsigned char *bytes = (unsigned char *)&val;
                int ret = _PyLong_AsByteArray((PyLongObject *)v,
                                              bytes, sizeof(val),
                                              is_little, !is_unsigned);
                Py_DECREF(v);
                if (likely(!ret))
                    return val;
            }
#endif
            return (long) -1;
        }
    } else {
        long val;
        PyObject *tmp = __Pyx_PyNumber_IntOrLong(x);
        if (!tmp) return (long) -1;
        val = __Pyx_PyInt_As_long(tmp);
        Py_DECREF(tmp);
        return val;
    }
raise_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "value too large to convert to long");
    return (long) -1;
raise_neg_overflow:
    PyErr_SetString(PyExc_OverflowError,
        "can't convert negative value to long");
    return (long) -1;
}

/* FastTypeChecks */
            #if CYTHON_COMPILING_IN_CPYTHON
static int __Pyx_InBases(PyTypeObject *a, PyTypeObject *b) {
    while (a) {
        a = a->tp_base;
        if (a == b)
            return 1;
    }
    return b == &PyBaseObject_Type;
}
static CYTHON_INLINE int __Pyx_IsSubtype(PyTypeObject *a, PyTypeObject *b) {
    PyObject *mro;
    if (a == b) return 1;
    mro = a->tp_mro;
    if (likely(mro)) {
        Py_ssize_t i, n;
        n = PyTuple_GET_SIZE(mro);
        for (i = 0; i < n; i++) {
            if (PyTuple_GET_ITEM(mro, i) == (PyObject *)b)
                return 1;
        }
        return 0;
    }
    return __Pyx_InBases(a, b);
}
#if PY_MAJOR_VERSION == 2
static int __Pyx_inner_PyErr_GivenExceptionMatches2(PyObject *err, PyObject* exc_type1, PyObject* exc_type2) {
    PyObject *exception, *value, *tb;
    int res;
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    __Pyx_ErrFetch(&exception, &value, &tb);
    res = exc_type1 ? PyObject_IsSubclass(err, exc_type1) : 0;
    if (unlikely(res == -1)) {
        PyErr_WriteUnraisable(err);
        res = 0;
    }
    if (!res) {
        res = PyObject_IsSubclass(err, exc_type2);
        if (unlikely(res == -1)) {
            PyErr_WriteUnraisable(err);
            res = 0;
        }
    }
    __Pyx_ErrRestore(exception, value, tb);
    return res;
}
#else
static CYTHON_INLINE int __Pyx_inner_PyErr_GivenExceptionMatches2(PyObject *err, PyObject* exc_type1, PyObject *exc_type2) {
    int res = exc_type1 ? __Pyx_IsSubtype((PyTypeObject*)err, (PyTypeObject*)exc_type1) : 0;
    if (!res) {
        res = __Pyx_IsSubtype((PyTypeObject*)err, (PyTypeObject*)exc_type2);
    }
    return res;
}
#endif
static int __Pyx_PyErr_GivenExceptionMatchesTuple(PyObject *exc_type, PyObject *tuple) {
    Py_ssize_t i, n;
    assert(PyExceptionClass_Check(exc_type));
    n = PyTuple_GET_SIZE(tuple);
#if PY_MAJOR_VERSION >= 3
    for (i=0; i<n; i++) {
        if (exc_type == PyTuple_GET_ITEM(tuple, i)) return 1;
    }
#endif
    for (i=0; i<n; i++) {
        PyObject *t = PyTuple_GET_ITEM(tuple, i);
        #if PY_MAJOR_VERSION < 3
        if (likely(exc_type == t)) return 1;
        #endif
        if (likely(PyExceptionClass_Check(t))) {
            if (__Pyx_inner_PyErr_GivenExceptionMatches2(exc_type, NULL, t)) return 1;
        } else {
        }
    }
    return 0;
}
static CYTHON_INLINE int __Pyx_PyErr_GivenExceptionMatches(PyObject *err, PyObject* exc_type) {
    if (likely(err == exc_type)) return 1;
    if (likely(PyExceptionClass_Check(err))) {
        if (likely(PyExceptionClass_Check(exc_type))) {
            return __Pyx_inner_PyErr_GivenExceptionMatches2(err, NULL, exc_type);
        } else if (likely(PyTuple_Check(exc_type))) {
            return __Pyx_PyErr_GivenExceptionMatchesTuple(err, exc_type);
        } else {
        }
    }
    return PyErr_GivenExceptionMatches(err, exc_type);
}
static CYTHON_INLINE int __Pyx_PyErr_GivenExceptionMatches2(PyObject *err, PyObject *exc_type1, PyObject *exc_type2) {
    assert(PyExceptionClass_Check(exc_type1));
    assert(PyExceptionClass_Check(exc_type2));
    if (likely(err == exc_type1 || err == exc_type2)) return 1;
    if (likely(PyExceptionClass_Check(err))) {
        return __Pyx_inner_PyErr_GivenExceptionMatches2(err, exc_type1, exc_type2);
    }
    return (PyErr_GivenExceptionMatches(err, exc_type1) || PyErr_GivenExceptionMatches(err, exc_type2));
}
#endif

/* CheckBinaryVersion */
            static int __Pyx_check_binary_version(void) {
    char ctversion[4], rtversion[4];
    PyOS_snprintf(ctversion, 4, "%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    PyOS_snprintf(rtversion, 4, "%s", Py_GetVersion());
    if (ctversion[0] != rtversion[0] || ctversion[2] != rtversion[2]) {
        char message[200];
        PyOS_snprintf(message, sizeof(message),
                      "compiletime version %s of module '%.100s' "
                      "does not match runtime version %s",
                      ctversion, __Pyx_MODULE_NAME, rtversion);
        return PyErr_WarnEx(NULL, message, 1);
    }
    return 0;
}

/* ModuleImport */
            #ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(const char *name) {
    PyObject *py_name = 0;
    PyObject *py_module = 0;
    py_name = __Pyx_PyIdentifier_FromString(name);
    if (!py_name)
        goto bad;
    py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    return py_module;
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

/* TypeImport */
            #ifndef __PYX_HAVE_RT_ImportType
#define __PYX_HAVE_RT_ImportType
static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name,
    size_t size, int strict)
{
    PyObject *py_module = 0;
    PyObject *result = 0;
    PyObject *py_name = 0;
    char warning[200];
    Py_ssize_t basicsize;
#ifdef Py_LIMITED_API
    PyObject *py_basicsize;
#endif
    py_module = __Pyx_ImportModule(module_name);
    if (!py_module)
        goto bad;
    py_name = __Pyx_PyIdentifier_FromString(class_name);
    if (!py_name)
        goto bad;
    result = PyObject_GetAttr(py_module, py_name);
    Py_DECREF(py_name);
    py_name = 0;
    Py_DECREF(py_module);
    py_module = 0;
    if (!result)
        goto bad;
    if (!PyType_Check(result)) {
        PyErr_Format(PyExc_TypeError,
            "%.200s.%.200s is not a type object",
            module_name, class_name);
        goto bad;
    }
#ifndef Py_LIMITED_API
    basicsize = ((PyTypeObject *)result)->tp_basicsize;
#else
    py_basicsize = PyObject_GetAttrString(result, "__basicsize__");
    if (!py_basicsize)
        goto bad;
    basicsize = PyLong_AsSsize_t(py_basicsize);
    Py_DECREF(py_basicsize);
    py_basicsize = 0;
    if (basicsize == (Py_ssize_t)-1 && PyErr_Occurred())
        goto bad;
#endif
    if (!strict && (size_t)basicsize > size) {
        PyOS_snprintf(warning, sizeof(warning),
            "%s.%s size changed, may indicate binary incompatibility. Expected %zd, got %zd",
            module_name, class_name, basicsize, size);
        if (PyErr_WarnEx(NULL, warning, 0) < 0) goto bad;
    }
    else if ((size_t)basicsize != size) {
        PyErr_Format(PyExc_ValueError,
            "%.200s.%.200s has the wrong size, try recompiling. Expected %zd, got %zd",
            module_name, class_name, basicsize, size);
        goto bad;
    }
    return (PyTypeObject *)result;
bad:
    Py_XDECREF(py_module);
    Py_XDECREF(result);
    return NULL;
}
#endif

/* InitStrings */
            static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        #if PY_MAJOR_VERSION < 3
        if (t->is_unicode) {
            *t->p = PyUnicode_DecodeUTF8(t->s, t->n - 1, NULL);
        } else if (t->intern) {
            *t->p = PyString_InternFromString(t->s);
        } else {
            *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        }
        #else
        if (t->is_unicode | t->is_str) {
            if (t->intern) {
                *t->p = PyUnicode_InternFromString(t->s);
            } else if (t->encoding) {
                *t->p = PyUnicode_Decode(t->s, t->n - 1, t->encoding, NULL);
            } else {
                *t->p = PyUnicode_FromStringAndSize(t->s, t->n - 1);
            }
        } else {
            *t->p = PyBytes_FromStringAndSize(t->s, t->n - 1);
        }
        #endif
        if (!*t->p)
            return -1;
        if (PyObject_Hash(*t->p) == -1)
            return -1;
        ++t;
    }
    return 0;
}

static CYTHON_INLINE PyObject* __Pyx_PyUnicode_FromString(const char* c_str) {
    return __Pyx_PyUnicode_FromStringAndSize(c_str, (Py_ssize_t)strlen(c_str));
}
static CYTHON_INLINE const char* __Pyx_PyObject_AsString(PyObject* o) {
    Py_ssize_t ignore;
    return __Pyx_PyObject_AsStringAndSize(o, &ignore);
}
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT
#if !CYTHON_PEP393_ENABLED
static const char* __Pyx_PyUnicode_AsStringAndSize(PyObject* o, Py_ssize_t *length) {
    char* defenc_c;
    PyObject* defenc = _PyUnicode_AsDefaultEncodedString(o, NULL);
    if (!defenc) return NULL;
    defenc_c = PyBytes_AS_STRING(defenc);
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
    {
        char* end = defenc_c + PyBytes_GET_SIZE(defenc);
        char* c;
        for (c = defenc_c; c < end; c++) {
            if ((unsigned char) (*c) >= 128) {
                PyUnicode_AsASCIIString(o);
                return NULL;
            }
        }
    }
#endif
    *length = PyBytes_GET_SIZE(defenc);
    return defenc_c;
}
#else
static CYTHON_INLINE const char* __Pyx_PyUnicode_AsStringAndSize(PyObject* o, Py_ssize_t *length) {
    if (unlikely(__Pyx_PyUnicode_READY(o) == -1)) return NULL;
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
    if (likely(PyUnicode_IS_ASCII(o))) {
        *length = PyUnicode_GET_LENGTH(o);
        return PyUnicode_AsUTF8(o);
    } else {
        PyUnicode_AsASCIIString(o);
        return NULL;
    }
#else
    return PyUnicode_AsUTF8AndSize(o, length);
#endif
}
#endif
#endif
static CYTHON_INLINE const char* __Pyx_PyObject_AsStringAndSize(PyObject* o, Py_ssize_t *length) {
#if __PYX_DEFAULT_STRING_ENCODING_IS_ASCII || __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT
    if (
#if PY_MAJOR_VERSION < 3 && __PYX_DEFAULT_STRING_ENCODING_IS_ASCII
            __Pyx_sys_getdefaultencoding_not_ascii &&
#endif
            PyUnicode_Check(o)) {
        return __Pyx_PyUnicode_AsStringAndSize(o, length);
    } else
#endif
#if (!CYTHON_COMPILING_IN_PYPY) || (defined(PyByteArray_AS_STRING) && defined(PyByteArray_GET_SIZE))
    if (PyByteArray_Check(o)) {
        *length = PyByteArray_GET_SIZE(o);
        return PyByteArray_AS_STRING(o);
    } else
#endif
    {
        char* result;
        int r = PyBytes_AsStringAndSize(o, &result, length);
        if (unlikely(r < 0)) {
            return NULL;
        } else {
            return result;
        }
    }
}
static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject* x) {
   int is_true = x == Py_True;
   if (is_true | (x == Py_False) | (x == Py_None)) return is_true;
   else return PyObject_IsTrue(x);
}
static PyObject* __Pyx_PyNumber_IntOrLongWrongResultType(PyObject* result, const char* type_name) {
#if PY_MAJOR_VERSION >= 3
    if (PyLong_Check(result)) {
        if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                "__int__ returned non-int (type %.200s).  "
                "The ability to return an instance of a strict subclass of int "
                "is deprecated, and may be removed in a future version of Python.",
                Py_TYPE(result)->tp_name)) {
            Py_DECREF(result);
            return NULL;
        }
        return result;
    }
#endif
    PyErr_Format(PyExc_TypeError,
                 "__%.4s__ returned non-%.4s (type %.200s)",
                 type_name, type_name, Py_TYPE(result)->tp_name);
    Py_DECREF(result);
    return NULL;
}
static CYTHON_INLINE PyObject* __Pyx_PyNumber_IntOrLong(PyObject* x) {
#if CYTHON_USE_TYPE_SLOTS
  PyNumberMethods *m;
#endif
  const char *name = NULL;
  PyObject *res = NULL;
#if PY_MAJOR_VERSION < 3
  if (likely(PyInt_Check(x) || PyLong_Check(x)))
#else
  if (likely(PyLong_Check(x)))
#endif
    return __Pyx_NewRef(x);
#if CYTHON_USE_TYPE_SLOTS
  m = Py_TYPE(x)->tp_as_number;
  #if PY_MAJOR_VERSION < 3
  if (m && m->nb_int) {
    name = "int";
    res = m->nb_int(x);
  }
  else if (m && m->nb_long) {
    name = "long";
    res = m->nb_long(x);
  }
  #else
  if (likely(m && m->nb_int)) {
    name = "int";
    res = m->nb_int(x);
  }
  #endif
#else
  if (!PyBytes_CheckExact(x) && !PyUnicode_CheckExact(x)) {
    res = PyNumber_Int(x);
  }
#endif
  if (likely(res)) {
#if PY_MAJOR_VERSION < 3
    if (unlikely(!PyInt_Check(res) && !PyLong_Check(res))) {
#else
    if (unlikely(!PyLong_CheckExact(res))) {
#endif
        return __Pyx_PyNumber_IntOrLongWrongResultType(res, name);
    }
  }
  else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "an integer is required");
  }
  return res;
}
static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject* b) {
  Py_ssize_t ival;
  PyObject *x;
#if PY_MAJOR_VERSION < 3
  if (likely(PyInt_CheckExact(b))) {
    if (sizeof(Py_ssize_t) >= sizeof(long))
        return PyInt_AS_LONG(b);
    else
        return PyInt_AsSsize_t(x);
  }
#endif
  if (likely(PyLong_CheckExact(b))) {
    #if CYTHON_USE_PYLONG_INTERNALS
    const digit* digits = ((PyLongObject*)b)->ob_digit;
    const Py_ssize_t size = Py_SIZE(b);
    if (likely(__Pyx_sst_abs(size) <= 1)) {
        ival = likely(size) ? digits[0] : 0;
        if (size == -1) ival = -ival;
        return ival;
    } else {
      switch (size) {
         case 2:
           if (8 * sizeof(Py_ssize_t) > 2 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -2:
           if (8 * sizeof(Py_ssize_t) > 2 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case 3:
           if (8 * sizeof(Py_ssize_t) > 3 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -3:
           if (8 * sizeof(Py_ssize_t) > 3 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((((size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case 4:
           if (8 * sizeof(Py_ssize_t) > 4 * PyLong_SHIFT) {
             return (Py_ssize_t) (((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
         case -4:
           if (8 * sizeof(Py_ssize_t) > 4 * PyLong_SHIFT) {
             return -(Py_ssize_t) (((((((((size_t)digits[3]) << PyLong_SHIFT) | (size_t)digits[2]) << PyLong_SHIFT) | (size_t)digits[1]) << PyLong_SHIFT) | (size_t)digits[0]));
           }
           break;
      }
    }
    #endif
    return PyLong_AsSsize_t(b);
  }
  x = PyNumber_Index(b);
  if (!x) return -1;
  ival = PyInt_AsSsize_t(x);
  Py_DECREF(x);
  return ival;
}
static CYTHON_INLINE PyObject * __Pyx_PyBool_FromLong(long b) {
  return b ? __Pyx_NewRef(Py_True) : __Pyx_NewRef(Py_False);
}
static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t ival) {
    return PyInt_FromSize_t(ival);
}


#endif /* Py_PYTHON_H */
