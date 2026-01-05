# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-src")
  file(MAKE_DIRECTORY "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-src")
endif()
file(MAKE_DIRECTORY
  "/tmp/tmpe99lfumb.build-temp/vllm-flash-attn"
  "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix"
  "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/tmp"
  "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp"
  "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src"
  "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/user1/easyhyum/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
