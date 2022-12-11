
find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE TEST_FILES "*_test.cpp")
add_executable(roboticssandbox-tests main.cpp ${TEST_FILES})