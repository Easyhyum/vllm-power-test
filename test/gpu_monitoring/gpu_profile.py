"""
Convolution Layer 기반 CUDA Graph 캡처 테스트 코드
"""
import torch
import torch.nn as nn
import time
import numpy as np
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("설치: pip install nvidia-ml-py")


class GPUMonitor:
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.handle = None
        self.metrics = []
        self.execution_time = None  # 실행 시간 (초)
        self.num_samples = None  # 처리된 샘플 수
        self.batch_size = None
        self.graph_batch_size = None
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                print(f"NVML 초기화 실패: {e}")
                self.handle = None
    
    def get_metrics(self):
        """현재 GPU 메트릭 수집"""
        if not NVML_AVAILABLE or self.handle is None:
            return None
        
        try:
            # 전력 (mW)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W로 변환
            
            # 온도 (°C)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK)

            # 그래픽 클럭 (MHz)
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            
            # 메모리 클럭 (MHz)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            
            # 메모리 사용량 (MB)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used = mem_info.used / (1024 ** 2)  # MB
            memory_total = mem_info.total / (1024 ** 2)  # MB
            
            # GPU 활용률 (utilization)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu
            memory_util = util.memory
            
            return {
                'power': power,
                'temperature': temp,
                'graphics_clock': graphics_clock,
                'memory_clock': memory_clock,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'gpu_util': gpu_util,
                'memory_util': memory_util
            }
        except Exception as e:
            print(f"메트릭 수집 오류: {e}")
            return None
    
    def start_monitoring(self, interval=0.01, duration=None):
        """모니터링 시작"""
        self.metrics = []
        if not NVML_AVAILABLE or self.handle is None:
            return
        
        start_time = time.time()
        while True:
            metric = self.get_metrics()
            if metric:
                metric['timestamp'] = time.time()
                self.metrics.append(metric)
            
            if duration and (time.time() - start_time) >= duration:
                break
            
            time.sleep(interval)
    
    def collect_during_execution(self, func, *args, num_samples=None, **kwargs):
        """함수 실행 중 메트릭 수집"""
        self.metrics = []
        self.num_samples = num_samples
        
        if not NVML_AVAILABLE or self.handle is None:
            start_time = time.time()
            result = func(*args, **kwargs)
            self.execution_time = time.time() - start_time
            return result
        
        import threading
        
        monitoring = True
        start_time = None
        
        def monitor_loop():
            nonlocal start_time
            start_time = time.time()
            while monitoring:
                metric = self.get_metrics()
                if metric:
                    metric['timestamp'] = time.time()
                    self.metrics.append(metric)
                time.sleep(0.01)  # 10ms 간격
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            monitoring = False
            monitor_thread.join(timeout=1.0)
            if start_time:
                self.execution_time = time.time() - start_time
        
        return result
    
    def get_statistics(self):
        """수집된 메트릭의 통계 계산"""
        if not self.metrics:
            return None
        
        stats = {}
        for key in ['power', 'temperature', 'graphics_clock', 'memory_clock', 
                   'gpu_util', 'memory_util']:
            values = [m[key] for m in self.metrics if key in m]
            if values:
                stats[key] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'current': values[-1] if values else None
                }
        
        return stats
    
    def calculate_energy_per_sample(self):
        """샘플당 에너지 (J/sample) 계산"""
        if not self.metrics or not self.execution_time or not self.num_samples:
            return None
        
        # 평균 전력 계산
        power_values = [m['power'] for m in self.metrics if 'power' in m]
        if not power_values:
            return None
        
        avg_power = sum(power_values) / len(power_values)  # W
        
        # 총 에너지 = 평균 전력 × 시간 (Joule)
        total_energy = avg_power * self.execution_time  # J
        
        # 샘플당 에너지
        energy_per_sample = total_energy / self.num_samples  # J/sample
        
        return {
            'total_energy': total_energy,
            'energy_per_sample': energy_per_sample,
            'avg_power': avg_power,
            'execution_time': self.execution_time,
            'num_samples': self.num_samples
        }
    
    def print_statistics(self, label=""):
        """통계 출력"""
        stats = self.get_statistics()
        if not stats:
            print(f"{label}: 메트릭 데이터가 없습니다.")
            return
        
        print(f"{label} GPU 메트릭:")
        print(f"  전력:")
        print(f"    평균: {stats['power']['avg']:.2f} W")
        print(f"    최소: {stats['power']['min']:.2f} W")
        print(f"    최대: {stats['power']['max']:.2f} W")
        print(f"    현재: {stats['power']['current']:.2f} W")
        
        # J/sample 계산 및 출력
        energy_info = self.calculate_energy_per_sample()
        if energy_info:
            print(f"  에너지 소비:")
            print(f"    총 에너지: {energy_info['total_energy']:.4f} J")
            print(f"    샘플당 에너지: {energy_info['energy_per_sample']:.6f} J/sample")
            print(f"    실행 시간: {energy_info['execution_time']:.4f} s")
            print(f"    처리 샘플 수: {energy_info['num_samples']:,}")
        
        print(f"  온도:")
        print(f"    평균: {stats['temperature']['avg']:.1f} °C")
        print(f"    최소: {stats['temperature']['min']:.1f} °C")
        print(f"    최대: {stats['temperature']['max']:.1f} °C")
        print(f"    현재: {stats['temperature']['current']:.1f} °C")
        
        print(f"  그래픽 클럭:")
        print(f"    평균: {stats['graphics_clock']['avg']:.0f} MHz")
        print(f"    최소: {stats['graphics_clock']['min']:.0f} MHz")
        print(f"    최대: {stats['graphics_clock']['max']:.0f} MHz")
        print(f"    현재: {stats['graphics_clock']['current']:.0f} MHz")
        
        print(f"  메모리 클럭:")
        print(f"    평균: {stats['memory_clock']['avg']:.0f} MHz")
        print(f"    최소: {stats['memory_clock']['min']:.0f} MHz")
        print(f"    최대: {stats['memory_clock']['max']:.0f} MHz")
        print(f"    현재: {stats['memory_clock']['current']:.0f} MHz")
        
        print(f"  GPU 활용률:")
        print(f"    평균: {stats['gpu_util']['avg']:.1f} %")
        print(f"    최소: {stats['gpu_util']['min']:.1f} %")
        print(f"    최대: {stats['gpu_util']['max']:.1f} %")
        print(f"    현재: {stats['gpu_util']['current']:.1f} %")
        
        print(f"  메모리 활용률:")
        print(f"    평균: {stats['memory_util']['avg']:.1f} %")
        print(f"    최소: {stats['memory_util']['min']:.1f} %")
        print(f"    최대: {stats['memory_util']['max']:.1f} %")
        print(f"    현재: {stats['memory_util']['current']:.1f} %")
        
        if self.metrics:
            mem_info = self.metrics[-1]
            if 'memory_used' in mem_info and 'memory_total' in mem_info:
                print(f"  메모리 사용량:")
                print(f"    사용: {mem_info['memory_used']:.0f} MB / {mem_info['memory_total']:.0f} MB")
                print(f"    사용률: {mem_info['memory_used'] / mem_info['memory_total'] * 100:.1f} %")


class ConvOnlyModel(nn.Module):
    """Convolution Layer만으로 구성된 모델"""
    
    def __init__(self, in_channels=3, num_layers=4, base_channels=64):
        super(ConvOnlyModel, self).__init__()
        
        layers = []
        channels = in_channels
        
        # 여러 개의 Convolution Layer 추가
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels = out_channels
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def benchmark_model(model, inputs, num_iterations=100, use_cuda_graph=False, monitor=None):
    """모델 성능 벤치마크"""
    model.eval()
    
    if use_cuda_graph:
        print("CUDA Graph를 사용한 벤치마크...")
    else:
        print("일반 실행 벤치마크...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)
    
    # 동기화
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 벤치마크 시작 (모니터링 포함)
    batch_size = inputs.shape[0]
    total_samples = batch_size * num_iterations
    
    def run_benchmark():
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        return (end_time - start_time) / num_iterations
    
    if monitor:
        avg_time = monitor.collect_during_execution(
            run_benchmark, num_samples=total_samples
        )
    else:
        avg_time = run_benchmark()
    
    throughput = batch_size / avg_time
    
    return avg_time, throughput


def main():
    """메인 함수"""
    print("=" * 60)
    print("Convolution Layer CUDA Graph 캡처 테스트")
    print("=" * 60)
    
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return
    
    device = torch.device("cuda")
    device_id = 0
    print(f"사용 중인 디바이스: {device}")
    print(f"GPU 이름: {torch.cuda.get_device_name(device_id)}")
    print(f"CUDA 버전: {torch.version.cuda}")
    if torch.backends.cudnn.is_available():
        print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
    
    # GPU 모니터 초기화
    monitor_normal = GPUMonitor(device_id=device_id)
    monitor_graph = GPUMonitor(device_id=device_id)
    if monitor_normal.handle:
        print("✓ GPU 모니터링 활성화")
    else:
        print("⚠ GPU 모니터링 비활성화 (NVML 사용 불가)")
    print()
    
    # 모델 및 입력 설정
    batch_size = 8
    in_channels = 3
    height = 224
    width = 224
    num_layers = 4
    base_channels = 64
    
    print(f"모델 설정:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input channels: {in_channels}")
    print(f"  - Input size: {height}x{width}")
    print(f"  - Convolution layers: {num_layers}")
    print(f"  - Base channels: {base_channels}")
    print()
    
    # 모델 생성 및 CUDA로 이동
    model = ConvOnlyModel(in_channels, num_layers, base_channels).to(device)
    model.eval()
    
    # 입력 데이터 생성 (seed 42로 고정)
    torch.manual_seed(42)
    np.random.seed(42)
    inputs = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    
    # CUDA Graph를 위한 static 텐서 생성 (같은 데이터 사용)
    # CUDA Graph는 static tensor를 요구하므로, 같은 데이터를 static tensor로 복사
    static_inputs = inputs.clone().detach()
    static_outputs = None  # forward 후 shape을 알 수 있으므로 나중에 생성
    
    # 입력 데이터가 동일한지 확인
    assert torch.allclose(inputs, static_inputs), "입력 데이터가 동일해야 합니다"
    
    print(f"입력 데이터 생성 완료 (seed=42)")
    print(f"일반 실행과 CUDA Graph가 같은 입력 데이터를 사용합니다.")
    print()
    
    print("=" * 60)
    print("1. 일반 실행 벤치마크")
    print("=" * 60)
    
    # 일반 실행 벤치마크 (모니터링 포함)
    avg_time_normal, throughput_normal = benchmark_model(
        model, inputs, num_iterations=100, use_cuda_graph=False, monitor=monitor_normal
    )
    
    print(f"평균 실행 시간: {avg_time_normal * 1000:.3f} ms")
    print(f"처리량: {throughput_normal:.2f} samples/sec")
    print()
    
    # 일반 실행 메트릭 출력
    monitor_normal.print_statistics("일반 실행")
    print()
    
    print("=" * 60)
    print("2. CUDA Graph 캡처 및 실행")
    print("=" * 60)
    
    # 출력 shape 확인을 위해 한 번 실행
    with torch.no_grad():
        sample_output = model(static_inputs)
        output_shape = sample_output.shape
        static_outputs = torch.empty_like(sample_output)
    
    print(f"출력 shape: {output_shape}")
    print()
    
    # Warmup 실행 (그래프 캡처 전에 최소 한 번 실행)
    with torch.no_grad():
        for _ in range(10):
            _ = model(static_inputs)
    
    torch.cuda.synchronize()
    
    # CUDA Graph 캡처
    print("CUDA Graph 캡처 중...")
    graph = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(graph):
        static_outputs = model(static_inputs)
    
    print("CUDA Graph 캡처 완료!")
    print()
    
    # CUDA Graph를 사용한 벤치마크 (모니터링 포함)
    print("CUDA Graph를 사용한 벤치마크...")
    torch.cuda.synchronize()
    
    num_iterations_graph = 100
    total_samples_graph = batch_size * num_iterations_graph
    
    def run_graph_benchmark():
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations_graph):
                graph.replay()
        torch.cuda.synchronize()
        end_time = time.time()
        return (end_time - start_time) / num_iterations_graph
    
    avg_time_graph = monitor_graph.collect_during_execution(
        run_graph_benchmark, num_samples=total_samples_graph
    )
    throughput_graph = batch_size / avg_time_graph
    
    print(f"평균 실행 시간: {avg_time_graph * 1000:.3f} ms")
    print(f"처리량: {throughput_graph:.2f} samples/sec")
    print()
    
    # CUDA Graph 실행 메트릭 출력
    monitor_graph.print_statistics("CUDA Graph 실행")
    print()
    
    print("=" * 60)
    print("성능 비교")
    print("=" * 60)
    speedup = avg_time_normal / avg_time_graph
    print(f"일반 실행: {avg_time_normal * 1000:.3f} ms")
    print(f"CUDA Graph: {avg_time_graph * 1000:.3f} ms")
    print(f"속도 향상: {speedup:.2f}x")
    if speedup > 1.0:
        print(f"✓ CUDA Graph가 {speedup:.2f}배 더 빠릅니다!")
    elif speedup < 1.0:
        print(f"⚠ 일반 실행이 {1.0/speedup:.2f}배 더 빠릅니다.")
    else:
        print("→ 성능이 거의 동일합니다.")
    print()
    
    # 에너지 효율 비교
    energy_normal = monitor_normal.calculate_energy_per_sample()
    energy_graph = monitor_graph.calculate_energy_per_sample()
    
    if energy_normal and energy_graph:
        print("=" * 60)
        print("에너지 효율 비교")
        print("=" * 60)
        print(f"일반 실행:")
        print(f"  샘플당 에너지: {energy_normal['energy_per_sample']:.6f} J/sample")
        print(f"  총 에너지: {energy_normal['total_energy']:.4f} J")
        print(f"CUDA Graph:")
        print(f"  샘플당 에너지: {energy_graph['energy_per_sample']:.6f} J/sample")
        print(f"  총 에너지: {energy_graph['total_energy']:.4f} J")
        
        energy_ratio = energy_normal['energy_per_sample'] / energy_graph['energy_per_sample']
        print(f"에너지 효율 비율: {energy_ratio:.2f}x")
        if energy_ratio > 1.0:
            print(f"✓ CUDA Graph가 {energy_ratio:.2f}배 더 에너지 효율적입니다!")
        elif energy_ratio < 1.0:
            print(f"⚠ 일반 실행이 {1.0/energy_ratio:.2f}배 더 에너지 효율적입니다.")
        else:
            print("→ 에너지 효율이 거의 동일합니다.")
        print()
    
    # 결과 검증 (같은 입력으로 비교)
    print("=" * 60)
    print("결과 검증")
    print("=" * 60)
    
    # 일반 실행 결과 (같은 입력 사용)
    with torch.no_grad():
        normal_output = model(inputs)
    
    # CUDA Graph 실행 결과 (같은 입력 데이터 사용)
    # static_inputs에 inputs의 값을 복사 (CUDA Graph는 static tensor를 요구)
    static_inputs.copy_(inputs)
    with torch.no_grad():
        graph.replay()
        graph_output = static_outputs.clone()
    
    # 결과 비교 (같은 입력을 사용했으므로 출력이 동일해야 함)
    max_diff = torch.max(torch.abs(normal_output - graph_output)).item()
    mean_diff = torch.mean(torch.abs(normal_output - graph_output)).item()
    rel_diff = torch.mean(torch.abs(normal_output - graph_output) / (torch.abs(normal_output) + 1e-8)).item()
    
    print("CUDA Graph 실행이 완료되었습니다.")
    print(f"일반 실행 출력 shape: {normal_output.shape}")
    print(f"CUDA Graph 출력 shape: {graph_output.shape}")
    print(f"최대 차이: {max_diff:.2e}")
    print(f"평균 차이: {mean_diff:.2e}")
    print(f"상대 차이: {rel_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ 결과가 일치합니다! (같은 입력 사용)")
    elif max_diff < 1e-3:
        print("⚠ 작은 수치 오차가 있습니다. (정상 범위)")
    else:
        print("⚠ 결과에 차이가 있습니다. 확인이 필요합니다.")
    print()
    
    # 모델 정보 출력
    print("=" * 60)
    print("모델 정보")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,}")
    print()
    
    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

