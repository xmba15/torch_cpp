UTEST=OFF
BUILD_EXAMPLES=OFF
BUILD_TYPE=Release
CMAKE_ARGS:=$(CMAKE_ARGS)
USE_GPU=OFF

default:
	@mkdir -p build
	@cd build && cmake .. -DBUILD_EXAMPLES=$(BUILD_EXAMPLES) \
	                      -DBUILD_TEST=$(UTEST) \
                              -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
                              -DUSE_GPU=$(USE_GPU) \
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                              $(CMAKE_ARGS)
	@cd build && make

debug:
	@make default BUILD_TYPE=Debug

apps:
	@make default BUILD_EXAMPLES=ON

gpu_apps:
	@make apps USE_GPU=ON

debug_apps:
	@make debug BUILD_EXAMPLES=ON

debug_gpu_apps:
	@make debug_apps USE_GPU=ON

unittest:
	@bash scripts/jit_superpoint_superglue_models.bash
	@make default UTEST=ON
	@cd build/tests && ./torch_cpp_unit_tests

clean:
	@rm -rf build*

install:
	@cd build && make install
