/*
 * CUDA Graph API - 통합 모듈
 * 
 * 기능:
 * 1. DOT 파일 저장 (cudaGraphDebugDotPrint)
 * 2. 그래프 노드 수정 (Grid dimension 변경 등)
 * 3. PyTorch CUDAGraph에서 내부 핸들 추출
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

// CUDA 에러 체크 매크로
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

// ============================================================
// DOT 파일 저장 기능
// ============================================================

// cudaGraphDebugDotFlags
enum GraphDebugDotFlags {
    DOT_FLAGS_DEFAULT = 0,
    DOT_FLAGS_VERBOSE = 1,
    DOT_FLAGS_RUNTIME_TYPES = 2,
    DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    DOT_FLAGS_HOST_NODE_PARAMS = 32,
    DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    DOT_FLAGS_HANDLES = 1024
};

/*
 * cudaGraph_t PyCapsule에서 DOT 파일 저장
 */
bool save_graph_dot_from_capsule(py::object graph_capsule, const std::string& filename, unsigned int flags) {
    if (!PyCapsule_CheckExact(graph_capsule.ptr())) {
        std::cerr << "Error: Expected a PyCapsule object" << std::endl;
        return false;
    }
    
    cudaGraph_t graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(graph_capsule.ptr(), nullptr));
    if (graph == nullptr) {
        std::cerr << "Error: Could not extract cudaGraph_t from capsule" << std::endl;
        return false;
    }
    
    cudaError_t err = cudaGraphDebugDotPrint(graph, filename.c_str(), flags);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "Graph saved to: " << filename << std::endl;
    return true;
}

/*
 * 스트림 캡처를 통해 그래프 생성 및 DOT 저장
 */
py::tuple capture_graph_and_save_dot(
    py::function forward_fn,
    py::object input_tensor,
    const std::string& filename,
    unsigned int flags
) {
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // PyTorch 스트림 설정
    auto torch_cuda = py::module_::import("torch").attr("cuda");
    
    // 그래프 캡처 시작
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    py::object output;
    try {
        output = forward_fn(input_tensor);
        torch_cuda.attr("synchronize")();
    } catch (const std::exception& e) {
        cudaStreamEndCapture(stream, &graph);
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("Error during capture: ") + e.what());
    }
    
    // 그래프 캡처 종료
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    
    // DOT 파일 저장
    if (!filename.empty()) {
        cudaError_t err = cudaGraphDebugDotPrint(graph, filename.c_str(), flags);
        if (err == cudaSuccess) {
            std::cout << "Graph saved to: " << filename << std::endl;
        }
    }
    
    // 그래프 인스턴스화
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    // 정리
    cudaStreamDestroy(stream);
    
    // PyCapsule로 반환
    py::capsule graph_capsule(graph, [](void* ptr) {
        if (ptr) cudaGraphDestroy(static_cast<cudaGraph_t>(ptr));
    });
    
    py::capsule exec_capsule(graphExec, [](void* ptr) {
        if (ptr) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(ptr));
    });
    
    return py::make_tuple(graph_capsule, exec_capsule, output);
}

/*
 * 빈 그래프 생성 후 DOT 저장 테스트
 */
bool test_empty_graph_dot(const std::string& filename) {
    cudaGraph_t graph;
    
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaGraphDebugDotPrint(graph, filename.c_str(), DOT_FLAGS_VERBOSE));
    
    std::cout << "Empty graph saved to: " << filename << std::endl;
    
    cudaGraphDestroy(graph);
    return true;
}

/*
 * PyTorch CUDAGraph 객체에서 DOT 파일 저장
 * 
 * PyTorch의 at::cuda::CUDAGraph 클래스를 직접 캐스팅하여 cudaGraph_t 접근
 */
bool save_pytorch_graph_dot(py::object cuda_graph_obj, const std::string& filename, unsigned int flags) {
    // 방법 1: PyTorch C++ 객체로 직접 캐스팅 (at::cuda::CUDAGraph)
    try {
        // torch.cuda.CUDAGraph -> at::cuda::CUDAGraph& 캐스팅
        at::cuda::CUDAGraph& cuda_graph = py::cast<at::cuda::CUDAGraph&>(cuda_graph_obj);
        
        // debug_dump() 호출 (내부적으로 cudaGraphDebugDotPrint 사용)
        cuda_graph.debug_dump(filename);
        
        // 파일 생성 확인
        std::ifstream f(filename);
        if (f.good()) {
            f.seekg(0, std::ios::end);
            size_t size = f.tellg();
            if (size > 0) {
                std::cout << "Saved via at::cuda::CUDAGraph::debug_dump(): " << filename 
                          << " (" << size << " bytes)" << std::endl;
                return true;
            }
        }
    } catch (const py::cast_error& e) {
        std::cerr << "py::cast to at::cuda::CUDAGraph failed: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "debug_dump() exception: " << e.what() << std::endl;
    }
    
    // 방법 2: Python debug_dump() 메서드 직접 호출 (fallback)
    if (py::hasattr(cuda_graph_obj, "debug_dump")) {
        try {
            cuda_graph_obj.attr("debug_dump")(filename);
            
            std::ifstream f(filename);
            if (f.good()) {
                f.seekg(0, std::ios::end);
                size_t size = f.tellg();
                if (size > 0) {
                    std::cout << "Saved via Python debug_dump(): " << filename 
                              << " (" << size << " bytes)" << std::endl;
                    return true;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Python debug_dump() exception: " << e.what() << std::endl;
        }
    }
    
    std::cerr << "Failed to save DOT file from PyTorch CUDAGraph" << std::endl;
    return false;
}

/*
 * PyTorch CUDAGraph 객체에서 노드 정보 추출
 */
py::list get_pytorch_graph_nodes_info(py::object cuda_graph_obj) {
    py::list nodes_info;
    
    try {
        at::cuda::CUDAGraph& cuda_graph = py::cast<at::cuda::CUDAGraph&>(cuda_graph_obj);
        std::cerr << "get_pytorch_graph_nodes_info: Direct node access not available" << std::endl;
    } catch (const py::cast_error& e) {
        std::cerr << "Cast error: " << e.what() << std::endl;
    }
    
    return nodes_info;
}

/*
 * PyTorch CUDAGraph 기본 정보 출력 및 DOT 파일 저장
 */
std::string print_pytorch_graph_info(py::object cuda_graph_obj, const std::string& filename) {
    std::stringstream ss;
    
    ss << "========================================\n";
    ss << "PyTorch CUDA Graph Info\n";
    ss << "========================================\n";
    
    try {
        at::cuda::CUDAGraph& cuda_graph = py::cast<at::cuda::CUDAGraph&>(cuda_graph_obj);
        
        ss << "Type: torch.cuda.CUDAGraph\n";
        ss << "Status: Captured\n";
        
        // pool 정보 확인
        if (py::hasattr(cuda_graph_obj, "pool")) {
            ss << "Memory Pool: Available\n";
        }
        
        // DOT 파일 저장 (debug_dump 메서드 사용)
        if (!filename.empty()) {
            try {
                cuda_graph.debug_dump(filename);
                
                // 파일 생성 확인
                std::ifstream f(filename);
                if (f.good()) {
                    f.seekg(0, std::ios::end);
                    size_t size = f.tellg();
                    ss << "DOT file saved: " << filename << " (" << size << " bytes)\n";
                } else {
                    ss << "DOT file may not have been created , filename: " << filename << std::endl;
                }
            } catch (const std::exception& e) {
                ss << "DOT save failed: " << e.what() << "\n";
            }
        }

        ss << "========================================\n";
        
    } catch (const py::cast_error& e) {
        ss << "Error: Could not cast to at::cuda::CUDAGraph\n";
        ss << "Details: " << e.what() << "\n";
    } catch (const std::exception& e) {
        ss << "Error: " << e.what() << "\n";
    }
    
    ss << "========================================\n";
    return ss.str();
}
/*
 * Llama-3.1-8B-Instruct 모델 전용 그래프 수정 함수
 * * 특징:
 * 1. Cutlass GEMM 커널 (Grid Y가 32 또는 48) 제외
 * 2. 일반 커널의 Grid X 차원 스케일링
 * 3. FlashAttention SplitKV 커널의 Grid Z 차원 스케일링
 */
 std::string manipulation_Llama_3_1_8B_Instruct_graph(
    py::object input_obj, 
    const int original_batch, 
    const int new_batch, 
    const std::string& filename
) {
    std::stringstream ss;

    ss << "========================================\n";
    ss << "Target Model: meta-llama/Llama-3.1-8B-Instruct\n";
    ss << "Manipulation: Batch " << original_batch << " -> " << new_batch << "\n";
    ss << "========================================\n";

    // 비율 계산
    float batch_ratio = static_cast<float>(new_batch) / static_cast<float>(original_batch);
    ss << "Batch Scale Ratio: " << batch_ratio << "\n";

    // 1. 디바이스 동기화 및 에러 초기화
    cudaDeviceSynchronize(); 
    cudaError_t prior_err = cudaGetLastError();
    if (prior_err != cudaSuccess) {
        ss << "[Warning] Cleared prior CUDA error: " << cudaGetErrorString(prior_err) << "\n";
    }

    // 2. 그래프 핸들 추출
    cudaGraph_t graph = nullptr;
    if (PyCapsule_CheckExact(input_obj.ptr())) {
        graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(input_obj.ptr(), nullptr));
    } else if (py::isinstance<py::int_>(input_obj)) {
        graph = reinterpret_cast<cudaGraph_t>(input_obj.cast<uint64_t>());
    } else {
        return "Error: Input must be PyCapsule or int";
    }

    if (graph == nullptr) return "Error: Invalid graph pointer";

    // 3. 노드 가져오기
    size_t numNodes = 0;
    cudaError_t err = cudaGraphGetNodes(graph, nullptr, &numNodes);
    if (err != cudaSuccess) return "CUDA Error: " + std::string(cudaGetErrorString(err));

    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);

    int modified_count = 0;
    int skipped_count = 0;

    // 4. 노드 순회 및 수정
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) continue;

        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            // 중요: memset으로 0 초기화 필수 (안하면 쓰레기값 들어감)
            memset(&params, 0, sizeof(params));
            
            err = cudaGraphKernelNodeGetParams(nodes[i], &params);
            if (err != cudaSuccess) {
                cudaGetLastError(); // 에러 클리어
                continue;
            }

            dim3 orig_grid = params.gridDim;
            dim3 new_grid = orig_grid;
            bool is_modified = false;
            std::string reason = "";

            // [Heuristic 1] Cutlass GEMM 보호
            // Llama 3.1 8B에서 Cutlass 커널은 Grid Y가 32 또는 48이며 배치 크기에 영향을 받지 않음
            if (orig_grid.y == 32 || orig_grid.y == 48) {
                skipped_count++;
                // ss << "  Node " << i << ": Skipped (Cutlass Pattern Y=" << orig_grid.y << ")\n";
                continue;
            }

            // [Heuristic 2] Grid X 스케일링 (일반적인 커널)
            // Grid X가 1보다 큰 경우 배치 비율에 맞춰 조정
            if (orig_grid.x > 1) {
                unsigned int scaled_x = static_cast<unsigned int>(round(orig_grid.x * batch_ratio));
                // 최소 1 보장 및 변경사항 확인
                if (scaled_x > 0 && scaled_x != orig_grid.x) {
                    new_grid.x = scaled_x;
                    is_modified = true;
                    reason += "Scale_X ";
                }
            }

            // [Heuristic 3] Grid Z 스케일링 (FlashAttention SplitKV 등)
            // Grid Z가 1보다 크고, Grid Y가 작은(헤드 그룹 등) 경우 Z축이 배치 차원일 가능성 높음
            if (orig_grid.z > 1 && orig_grid.y < 16) {
                unsigned int scaled_z = static_cast<unsigned int>(round(orig_grid.z * batch_ratio));
                if (scaled_z > 0 && scaled_z != orig_grid.z) {
                    new_grid.z = scaled_z;
                    is_modified = true;
                    reason += "Scale_Z ";
                }
            }

            // 수정 적용
            if (is_modified) {
                params.gridDim = new_grid;
                err = cudaGraphKernelNodeSetParams(nodes[i], &params);
                if (err == cudaSuccess) {
                    ss << "  Node " << i << " [" << reason << "]: (" 
                       << orig_grid.x << "," << orig_grid.y << "," << orig_grid.z << ") -> ("
                       << new_grid.x << "," << new_grid.y << "," << new_grid.z << ")\n";
                    modified_count++;
                } else {
                    ss << "  Node " << i << ": Failed update: " << cudaGetErrorString(err) << "\n";
                }
            }
        }
    }

    // 5. 결과 저장 (선택)
    if (!filename.empty()) {
        cudaGraphDebugDotPrint(graph, filename.c_str(), DOT_FLAGS_VERBOSE);
    }

    ss << "----------------------------------------\n";
    ss << "Modified: " << modified_count << " nodes\n";
    ss << "Skipped (Cutlass): " << skipped_count << " nodes\n";
    ss << "========================================\n";

    // 종료 전 정리
    cudaDeviceSynchronize();
    cudaGetLastError();

    return ss.str();
}

std::string manipulation_pytorch_graph(py::object input_obj, const int original_batch, const int new_batch, const std::string& filename) {
    std::stringstream ss;

    ss << "========================================\n";
    ss << "PyTorch CUDA Graph Manipulation: " << original_batch << " -> " << new_batch << "\n";
    ss << "========================================\n";

    float batch_ratio = static_cast<float>(new_batch) / static_cast<float>(original_batch);

    // [핵심 수정 1] 진입 전 디바이스 동기화 및 기존 에러 클리어
    // 이전 단계(debug_dump 등)에서 발생한 비동기 에러가 있다면 여기서 잡아서 리셋합니다.
    cudaDeviceSynchronize(); 
    cudaError_t prior_err = cudaGetLastError();
    if (prior_err != cudaSuccess) {
        ss << "[Warning] Found and cleared prior CUDA error: " << cudaGetErrorString(prior_err) << "\n";
    }

    cudaGraph_t graph = nullptr;

    // Case 1: PyCapsule (구버전 호환)
    if (PyCapsule_CheckExact(input_obj.ptr())) {
        graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(input_obj.ptr(), nullptr));
        ss << "Input Type: PyCapsule\n";
    } 
    // Case 2: Integer (현재 PyTorch 버전)
    else if (py::isinstance<py::int_>(input_obj)) {
        uint64_t ptr_val = input_obj.cast<uint64_t>();
        graph = reinterpret_cast<cudaGraph_t>(ptr_val);
        ss << "Input Type: Integer (Address: " << ptr_val << ")\n";
    }
    else {
        return "Error: Expected PyCapsule or int, got " + std::string(py::str(input_obj.get_type()));
    }

    if (graph == nullptr) {
        return "Error: Invalid graph pointer (nullptr)";
    }

    size_t numNodes = 0;
    cudaError_t err = cudaGraphGetNodes(graph, nullptr, &numNodes);
    
    if (err != cudaSuccess) {
        // 여기서 에러가 난다면 진짜 핸들 문제거나 심각한 드라이버 오류입니다.
        return "CUDA Error (cudaGraphGetNodes): " + std::string(cudaGetErrorString(err));
    }

    std::vector<cudaGraphNode_t> nodes(numNodes);
    err = cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    if (err != cudaSuccess) {
        return "CUDA Error (get node list): " + std::string(cudaGetErrorString(err));
    }

    int kernel_count = 0;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType type;
        err = cudaGraphNodeGetType(nodes[i], &type);
        if (err != cudaSuccess) continue;

        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            memset(&params, 0, sizeof(params));
            err = cudaGraphKernelNodeGetParams(nodes[i], &params);

            dim3 orig_grid = params.gridDim;

            if (err == cudaSuccess) {
                
                params.gridDim.x = static_cast<unsigned int>(orig_grid.x * batch_ratio);
                err = cudaGraphKernelNodeSetParams(nodes[i], &params);
                if (err != cudaSuccess) {
                    ss << "  Node " << i << ": Failed to set params: " << cudaGetErrorString(err) << "\n";
                } else {
                    ss << "  Node " << i << ": Grid (" << orig_grid.x << ", " 
                       << orig_grid.y << ", " << orig_grid.z << ") -> ("
                       << params.gridDim.x << ", " 
                       << params.gridDim.y << ", " << params.gridDim.z << ")\n";
                }
            } else {
                ss << "  Node " << i << ": KERNEL (params unavailable)\n";
                kernel_count++;
                // 에러 클리어 - 다음 노드 진행을 위해
                cudaGetLastError();
            }

        }
    }

    try{
        CUDA_CHECK(cudaGraphDebugDotPrint(graph, filename.c_str(), DOT_FLAGS_VERBOSE));
    } catch (const std::exception& e) {
        ss << "DOT save failed: "<< filename << ": " << e.what() << "\n";
    }

    ss << "  Total Kernel Nodes: " << kernel_count << "\n";
    ss << "========================================\n";
    
    // [중요] 함수 종료 전 모든 에러 상태 클리어
    cudaDeviceSynchronize();
    cudaGetLastError();  // 에러 상태 리셋
    
    return ss.str();
}

std::string cuda_graph_debug_print(py::object input_obj, const std::string& filename) {
    std::stringstream ss;

    ss << "========================================\n";
    ss << "PyTorch CUDA Graph Debug Print\n";
    ss << "========================================\n";

    // [핵심 수정 1] 진입 전 디바이스 동기화 및 기존 에러 클리어
    // 이전 단계(debug_dump 등)에서 발생한 비동기 에러가 있다면 여기서 잡아서 리셋합니다.
    cudaDeviceSynchronize(); 
    cudaError_t prior_err = cudaGetLastError();
    if (prior_err != cudaSuccess) {
        ss << "[Warning] Found and cleared prior CUDA error: " << cudaGetErrorString(prior_err) << "\n";
    }

    cudaGraph_t graph = nullptr;

    // Case 1: PyCapsule (구버전 호환)
    if (PyCapsule_CheckExact(input_obj.ptr())) {
        graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(input_obj.ptr(), nullptr));
        ss << "Input Type: PyCapsule\n";
    } 
    // Case 2: Integer (현재 PyTorch 버전)
    else if (py::isinstance<py::int_>(input_obj)) {
        uint64_t ptr_val = input_obj.cast<uint64_t>();
        graph = reinterpret_cast<cudaGraph_t>(ptr_val);
        ss << "Input Type: Integer (Address: " << ptr_val << ")\n";
    }
    else {
        return "Error: Expected PyCapsule or int, got " + std::string(py::str(input_obj.get_type()));
    }

    if (graph == nullptr) {
        return "Error: Invalid graph pointer (nullptr)";
    }

    try{
        CUDA_CHECK(cudaGraphDebugDotPrint(graph, filename.c_str(), DOT_FLAGS_VERBOSE));
    } catch (const std::exception& e) {
        ss << "DOT save failed: "<< filename << ": " << e.what() << "\n";
    }

    size_t numNodes = 0;
    cudaError_t err = cudaGraphGetNodes(graph, nullptr, &numNodes);
    
    if (err != cudaSuccess) {
        // 여기서 에러가 난다면 진짜 핸들 문제거나 심각한 드라이버 오류입니다.
        return "CUDA Error (cudaGraphGetNodes): " + std::string(cudaGetErrorString(err));
    }

    std::vector<cudaGraphNode_t> nodes(numNodes);
    err = cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    if (err != cudaSuccess) {
        return "CUDA Error (get node list): " + std::string(cudaGetErrorString(err));
    }

    int kernel_count = 0;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType type;
        err = cudaGraphNodeGetType(nodes[i], &type);
        if (err != cudaSuccess) continue;

        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            memset(&params, 0, sizeof(params));
            err = cudaGraphKernelNodeGetParams(nodes[i], &params);
            if (err == cudaSuccess) {
                ss << "  Node " << i << ": Grid (" << params.gridDim.x << ", " 
                   << params.gridDim.y << ", " << params.gridDim.z << ")\n";
                kernel_count++;
            } else {
                ss << "  Node " << i << ": KERNEL (params unavailable)\n";
                kernel_count++;
                // 에러 클리어 - 다음 노드 진행을 위해
                cudaGetLastError();
            }
        }
    }
    
    ss << "  Total Kernel Nodes: " << kernel_count << "\n";
    ss << "========================================\n";
    
    // [중요] 함수 종료 전 모든 에러 상태 클리어
    cudaDeviceSynchronize();
    cudaGetLastError();  // 에러 상태 리셋
    
    return ss.str();
}


// ============================================================
// 그래프 노드 조작 기능
// ============================================================

// 노드 타입 문자열 변환
std::string node_type_to_string(cudaGraphNodeType type) {
    switch (type) {
        case cudaGraphNodeTypeKernel: return "KERNEL";
        case cudaGraphNodeTypeMemcpy: return "MEMCPY";
        case cudaGraphNodeTypeMemset: return "MEMSET";
        case cudaGraphNodeTypeHost: return "HOST";
        case cudaGraphNodeTypeGraph: return "CHILD_GRAPH";
        case cudaGraphNodeTypeEmpty: return "EMPTY";
        case cudaGraphNodeTypeWaitEvent: return "WAIT_EVENT";
        case cudaGraphNodeTypeEventRecord: return "EVENT_RECORD";
        default: return "UNKNOWN";
    }
}

/*
 * 그래프 노드 정보를 딕셔너리 리스트로 반환
 */
py::list get_graph_nodes_info(py::object graph_capsule) {
    py::list nodes_info;
    
    if (!PyCapsule_CheckExact(graph_capsule.ptr())) {
        throw std::runtime_error("Expected a PyCapsule object");
    }
    
    cudaGraph_t graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(graph_capsule.ptr(), nullptr));
    if (graph == nullptr) {
        throw std::runtime_error("Could not extract cudaGraph_t from capsule");
    }
    
    size_t numNodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
    
    std::vector<cudaGraphNode_t> nodes(numNodes);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
    
    for (size_t i = 0; i < numNodes; i++) {
        py::dict node_info;
        node_info["index"] = i;
        
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
        node_info["type"] = node_type_to_string(type);
        
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[i], &params));
            
            py::dict grid;
            grid["x"] = params.gridDim.x;
            grid["y"] = params.gridDim.y;
            grid["z"] = params.gridDim.z;
            node_info["grid"] = grid;
            
            py::dict block;
            block["x"] = params.blockDim.x;
            block["y"] = params.blockDim.y;
            block["z"] = params.blockDim.z;
            node_info["block"] = block;
            
            node_info["shared_memory"] = params.sharedMemBytes;
            node_info["func_ptr"] = reinterpret_cast<uint64_t>(params.func);
            
        } else if (type == cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms memcpyParams = {0};
            CUDA_CHECK(cudaGraphMemcpyNodeGetParams(nodes[i], &memcpyParams));
            
            std::string kind;
            switch (memcpyParams.kind) {
                case cudaMemcpyHostToDevice: kind = "HostToDevice"; break;
                case cudaMemcpyDeviceToHost: kind = "DeviceToHost"; break;
                case cudaMemcpyDeviceToDevice: kind = "DeviceToDevice"; break;
                default: kind = "Other"; break;
            }
            node_info["memcpy_kind"] = kind;
            
            py::dict extent;
            extent["width"] = memcpyParams.extent.width;
            extent["height"] = memcpyParams.extent.height;
            extent["depth"] = memcpyParams.extent.depth;
            node_info["extent"] = extent;
            
        } else if (type == cudaGraphNodeTypeMemset) {
            cudaMemsetParams memsetParams;
            CUDA_CHECK(cudaGraphMemsetNodeGetParams(nodes[i], &memsetParams));
            
            node_info["value"] = memsetParams.value;
            node_info["width"] = memsetParams.width;
            node_info["height"] = memsetParams.height;
        }
        
        nodes_info.append(node_info);
    }
    
    return nodes_info;
}

/*
 * 그래프의 커널 노드 Grid dimension 수정
 */
bool modify_kernel_node_grid(
    py::object graph_capsule,
    int node_index,
    int grid_x, int grid_y, int grid_z
) {
    if (!PyCapsule_CheckExact(graph_capsule.ptr())) {
        throw std::runtime_error("Expected a PyCapsule object");
    }
    
    cudaGraph_t graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(graph_capsule.ptr(), nullptr));
    if (graph == nullptr) {
        throw std::runtime_error("Could not extract cudaGraph_t from capsule");
    }
    
    size_t numNodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
    
    if (node_index < 0 || node_index >= static_cast<int>(numNodes)) {
        throw std::runtime_error("Node index out of range");
    }
    
    std::vector<cudaGraphNode_t> nodes(numNodes);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
    
    cudaGraphNodeType type;
    CUDA_CHECK(cudaGraphNodeGetType(nodes[node_index], &type));
    
    if (type != cudaGraphNodeTypeKernel) {
        throw std::runtime_error("Node is not a kernel node");
    }
    
    cudaKernelNodeParams params;
    CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[node_index], &params));
    
    if (grid_x >= 0) params.gridDim.x = grid_x;
    if (grid_y >= 0) params.gridDim.y = grid_y;
    if (grid_z >= 0) params.gridDim.z = grid_z;
    
    CUDA_CHECK(cudaGraphKernelNodeSetParams(nodes[node_index], &params));
    
    return true;
}

/*
 * Batch 비율에 따라 모든 커널 노드의 Grid dimension 자동 수정
 */
int modify_graph_for_batch(
    py::object graph_capsule,
    int original_batch,
    int new_batch
) {
    if (!PyCapsule_CheckExact(graph_capsule.ptr())) {
        throw std::runtime_error("Expected a PyCapsule object");
    }
    
    cudaGraph_t graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(graph_capsule.ptr(), nullptr));
    if (graph == nullptr) {
        throw std::runtime_error("Could not extract cudaGraph_t from capsule");
    }
    
    size_t numNodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
    
    std::vector<cudaGraphNode_t> nodes(numNodes);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
    
    float batch_ratio = static_cast<float>(new_batch) / static_cast<float>(original_batch);
    int modified_count = 0;
    
    std::cout << "Modifying graph: Batch " << original_batch << " -> " << new_batch << std::endl;
    std::cout << "Batch ratio: " << batch_ratio << std::endl;
    
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
        
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[i], &params));
            
            dim3 orig_grid = params.gridDim;
            bool modified = false;
            
            // 패턴: batch에 비례하는 차원 스케일링
            if (orig_grid.z > 1 && orig_grid.z == static_cast<unsigned int>(original_batch * 8)) {
                params.gridDim.z = static_cast<unsigned int>(orig_grid.z * batch_ratio);
                modified = true;
            }
            else if (orig_grid.y > 1 && orig_grid.y == static_cast<unsigned int>(original_batch * 2)) {
                params.gridDim.y = static_cast<unsigned int>(orig_grid.y * batch_ratio);
                modified = true;
            }
            else if (orig_grid.x > 1 && orig_grid.y == 1 && orig_grid.z == 1) {
                params.gridDim.x = static_cast<unsigned int>(std::max(1.0f, orig_grid.x * batch_ratio));
                modified = true;
            }
            
            if (modified) {
                CUDA_CHECK(cudaGraphKernelNodeSetParams(nodes[i], &params));
                
                std::cout << "  Node " << i << ": Grid (" 
                          << orig_grid.x << "," << orig_grid.y << "," << orig_grid.z << ") -> ("
                          << params.gridDim.x << "," << params.gridDim.y << "," << params.gridDim.z << ")" << std::endl;
                modified_count++;
            }
        }
    }
    
    std::cout << "Modified " << modified_count << " kernel nodes" << std::endl;
    return modified_count;
}

/*
 * 수정된 그래프를 재인스턴스화
 */
py::object reinstantiate_graph(py::object graph_capsule) {
    if (!PyCapsule_CheckExact(graph_capsule.ptr())) {
        throw std::runtime_error("Expected a PyCapsule object");
    }
    
    cudaGraph_t graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(graph_capsule.ptr(), nullptr));
    if (graph == nullptr) {
        throw std::runtime_error("Could not extract cudaGraph_t from capsule");
    }
    
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    return py::capsule(graphExec, [](void* ptr) {
        if (ptr) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(ptr));
    });
}

/*
 * 그래프 노드 개수 반환
 */
int64_t get_node_count(py::object graph_capsule) {
    if (!PyCapsule_CheckExact(graph_capsule.ptr())) {
        return -1;
    }
    
    cudaGraph_t graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(graph_capsule.ptr(), nullptr));
    if (graph == nullptr) {
        return -1;
    }
    
    size_t numNodes;
    if (cudaGraphGetNodes(graph, nullptr, &numNodes) != cudaSuccess) {
        return -1;
    }
    
    return static_cast<int64_t>(numNodes);
}

/*
 * 그래프 정보를 문자열로 출력
 */
std::string print_graph_info(py::object graph_capsule) {
    std::stringstream ss;
    
    if (!PyCapsule_CheckExact(graph_capsule.ptr())) {
        return "Error: Not a valid PyCapsule";
    }
    
    cudaGraph_t graph = static_cast<cudaGraph_t>(PyCapsule_GetPointer(graph_capsule.ptr(), nullptr));
    if (graph == nullptr) {
        return "Error: Could not extract cudaGraph_t";
    }
    
    size_t numNodes;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
    
    std::vector<cudaGraphNode_t> nodes(numNodes);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
    
    ss << "========================================\n";
    ss << "Graph Information\n";
    ss << "Total Nodes: " << numNodes << "\n";
    ss << "========================================\n";
    
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
        
        ss << "\nNode " << i << ": " << node_type_to_string(type) << "\n";
        
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[i], &params));
            
            ss << "  Grid: (" << params.gridDim.x << ", " 
               << params.gridDim.y << ", " << params.gridDim.z << ")\n";
            ss << "  Block: (" << params.blockDim.x << ", " 
               << params.blockDim.y << ", " << params.blockDim.z << ")\n";
            ss << "  Shared Memory: " << params.sharedMemBytes << " bytes\n";
        }
    }
    
    ss << "========================================\n";
    return ss.str();
}

// ============================================================
// Python 모듈 정의
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA Graph API - DOT file saving and graph manipulation";
    
    // DOT 저장 기능
    m.def("save_graph_dot_from_capsule", &save_graph_dot_from_capsule,
          "Save CUDA graph to DOT file from PyCapsule",
          py::arg("graph_capsule"),
          py::arg("filename"),
          py::arg("flags") = 1u);  // DOT_FLAGS_VERBOSE = 1
    
    m.def("capture_graph_and_save_dot", &capture_graph_and_save_dot,
          "Capture model execution, save DOT file, and return graph handles",
          py::arg("forward_fn"),
          py::arg("input_tensor"),
          py::arg("filename") = "",
          py::arg("flags") = 1u);  // DOT_FLAGS_VERBOSE = 1
    
    m.def("test_empty_graph_dot", &test_empty_graph_dot,
          "Test by creating and saving an empty graph",
          py::arg("filename"));
    
    m.def("save_pytorch_graph_dot", &save_pytorch_graph_dot,
          "Save PyTorch CUDAGraph to DOT file (direct access to at::cuda::CUDAGraph)",
          py::arg("cuda_graph"),
          py::arg("filename"),
          py::arg("flags") = 1u);  // DOT_FLAGS_VERBOSE = 1
    
    m.def("get_pytorch_graph_nodes_info", &get_pytorch_graph_nodes_info,
          "Get node info from PyTorch CUDAGraph",
          py::arg("cuda_graph"));
    
    m.def("print_pytorch_graph_info", &print_pytorch_graph_info,
          "Print basic info of PyTorch CUDAGraph and save DOT file",
          py::arg("cuda_graph"),
          py::arg("filename") = "");
          
    m.def("manipulation_pytorch_graph", &manipulation_pytorch_graph,
    "PyTorch CUDAGraph debug print",
        py::arg("graph_capsule"),
        py::arg("original_batch"),
        py::arg("new_batch"),
        py::arg("filename") = "");

    m.def("manipulation_Llama_3_1_8B_Instruct_graph", &manipulation_Llama_3_1_8B_Instruct_graph,
        "PyTorch CUDAGraph debug print",
            py::arg("graph_capsule"),
            py::arg("original_batch"),
            py::arg("new_batch"),
            py::arg("filename") = "");
            
    m.def("cuda_graph_debug_print", &cuda_graph_debug_print,
          "PyTorch CUDAGraph debug print",
          py::arg("cuda_graph"),
          py::arg("filename") = "");
    
    // 그래프 조작 기능
    m.def("get_graph_nodes_info", &get_graph_nodes_info,
          "Get list of node information dictionaries",
          py::arg("graph_capsule"));
    
    m.def("modify_kernel_node_grid", &modify_kernel_node_grid,
          "Modify kernel node grid dimensions (-1 to keep original)",
          py::arg("graph_capsule"),
          py::arg("node_index"),
          py::arg("grid_x") = -1,
          py::arg("grid_y") = -1,
          py::arg("grid_z") = -1);
    
    m.def("modify_graph_for_batch", &modify_graph_for_batch,
          "Auto-modify all kernel nodes for new batch size",
          py::arg("graph_capsule"),
          py::arg("original_batch"),
          py::arg("new_batch"));
    
    m.def("reinstantiate_graph", &reinstantiate_graph,
          "Create new GraphExec from modified graph",
          py::arg("graph_capsule"));
    
    m.def("get_node_count", &get_node_count,
          "Get number of nodes in graph",
          py::arg("graph_capsule"));
    
    m.def("print_graph_info", &print_graph_info,
          "Get formatted string of graph information",
          py::arg("graph_capsule"));
    
    // DOT 플래그 상수 (정수 리터럴로 직접 정의)
    m.attr("DOT_FLAGS_DEFAULT") = 0u;
    m.attr("DOT_FLAGS_VERBOSE") = 1u;
    m.attr("DOT_FLAGS_RUNTIME_TYPES") = 2u;
    m.attr("DOT_FLAGS_KERNEL_NODE_PARAMS") = 4u;
    m.attr("DOT_FLAGS_MEMCPY_NODE_PARAMS") = 8u;
    m.attr("DOT_FLAGS_MEMSET_NODE_PARAMS") = 16u;
    m.attr("DOT_FLAGS_HANDLES") = 1024u;
}
