INCLUDE_DIRECTORIES(.)
FILE(GLOB hdrs "*.h*")
FILE(GLOB srcs "*.c*")


SET(hdrs ${hdrs})
SET(srcs ${srcs})

# ----------------------------------------------------------------------------
#   CUDA
# ----------------------------------------------------------------------------
FIND_PACKAGE(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_60;-lineinfo;--compiler-options;-O2")
SET(CUDA_PROPAGATE_HOST_FLAGS ON)
set(cuda_src cuda_aruco.cu cudaImage.cu)
cuda_add_library(cuda_aruco ${cuda_src})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

ADD_LIBRARY(${PROJECT_NAME} ${srcs} ${hdrs})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR=${OpenCV_LIB_DIR}")
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES 				# create *nix style library versions + symbolic links
	DEFINE_SYMBOL DSO_EXPORTS
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_SOVERSION}
	CLEAN_DIRECT_OUTPUT 1 					# allow creating static and shared libs without conflicts
	OUTPUT_NAME "${PROJECT_NAME}${PROJECT_DLLVERSION}" 	# avoid conflicts between library and binary target names
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${REQUIRED_LIBRARIES} cuda_aruco)

INSTALL(TARGETS ${PROJECT_NAME} cuda_aruco
	RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
	LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
	ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory

INSTALL(FILES ${hdrs}
	DESTINATION include/${PROJECT_NAME}
	COMPONENT main)

