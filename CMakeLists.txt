cmake_minimum_required(VERSION 3.6)
project(ExecutableProject)

set(CMAKE_C_STANDARD 99)

file(GLOB SOURCE_FILES include/*.c)

add_executable(HAEASequential ${SOURCE_FILES} HAEASequential.c)
add_executable(HAEAPosix ${SOURCE_FILES} HAEAPosix.c)
add_executable(Probe ${SOURCE_FILES} Probe.c)