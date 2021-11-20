BUILD_EXAMPLES=OFF
BUILD_TYPE=Release
CMAKE_ARGS:=$(CMAKE_ARGS)
USE_GPU=OFF

default:
	@mkdir -p build
	@cd build && cmake .. -DBUILD_EXAMPLES=$(BUILD_EXAMPLES) \
                              -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
                              -DUSE_GPU=$(USE_GPU) \
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                              $(CMAKE_ARGS)
	@cd build && make

debug:
	@make default BUILD_TYPE=Debug

apps:
	@make default BUIILD_EXAMPLES=ON

debug_apps:
	@make debug BUILD_EXAMPLES=ON

clean:
	@rm -rf build*

install:
	@cd build && make install
