const std = @import("std");
const Complex = @import("Complex.zig").Complex;

/// S4 (Structured State Space) Layer: Handles multi-channel sequence modeling
/// via parallelizable convolutional representations.
pub const S4Layer = struct {
    allocator: std.mem.Allocator,
    dt: f32, // Step size for discretization
    a_bars: []Complex, // Discretized state transition parameters
    b_bars: []Complex, // Discretized input matrix parameters
    c_coeffs: []Complex, // Output projection coefficients
    states: []Complex, // Latent state vectors (for recurrent mode)
    kernels: [][]Complex, // Pre-computed SSM Convolutional Kernels
    temp_buffer: []Complex, // Intermediate scratchpad for per-channel convolution
    fft_input_buffer: []Complex, // N 크기의 고정 버퍼 (init에서 할당)
    fft_result_buffer: []Complex, // 커널 FFT 결과나 ERROR 신호 FFT 결과 저장
    output_buffer: []Complex, // Final aggregated output buffer
    a_continuous: []Complex,
    b_continuous: []Complex,

    fn getFFTLength(seqLen: usize, kernelLen: usize) !usize {
        // 1. 필요한 N 크기 결정 Linear Convolution(선형 합성곱)
        // RNN과 CNN의 결과가 다른 범인은 바로 **"선형 컨볼루션 vs 순환 컨볼루션"**의 차이
        // 패딩 크기 N이 최소한 입력 길이 + 커널 길이 - 1 보다는 커야 합니다.
        const required_len = seqLen + kernelLen;
        const n = try std.math.ceilPowerOfTwo(usize, required_len);
        return n;
    }

    pub fn init(allocator: std.mem.Allocator, numChannels: usize, kernelLen: usize, inputLen: usize, dt: f32, A: []const Complex, B: []const Complex, C: []const Complex) !*S4Layer {
        const self = try allocator.create(S4Layer);
        errdefer self.deinit();

        // Initialize struct fields to zero/empty
        self.* = .{
            .allocator = allocator,
            .dt = dt,
            .a_bars = &[_]Complex{},
            .b_bars = &[_]Complex{},
            .c_coeffs = &[_]Complex{},
            .states = &[_]Complex{},
            .kernels = &[_][]Complex{},
            .temp_buffer = &[_]Complex{},
            .fft_input_buffer = &[_]Complex{},
            .fft_result_buffer = &[_]Complex{},
            .output_buffer = &[_]Complex{},
            .a_continuous = &[_]Complex{},
            .b_continuous = &[_]Complex{},
        };

        // Memory allocation for SSM parameters and latent states
        self.a_bars = try allocator.alloc(Complex, numChannels);
        self.b_bars = try allocator.alloc(Complex, numChannels);
        self.c_coeffs = try allocator.alloc(Complex, numChannels);
        self.states = try allocator.alloc(Complex, numChannels);
        @memset(self.states, Complex.init(0, 0));

        // Memory allocation for pre-computed Convolutional Kernels (K = C * A^i * B)
        self.kernels = try allocator.alloc([]Complex, numChannels);
        @memset(self.kernels, &[_]Complex{});
        for (self.kernels) |*k| {
            k.* = try allocator.alloc(Complex, kernelLen);
        }

        // Buffers for vectorized signal processing
        self.temp_buffer = try allocator.alloc(Complex, inputLen);
        @memset(self.temp_buffer, Complex.init(0, 0));
        self.output_buffer = try allocator.alloc(Complex, inputLen);
        @memset(self.output_buffer, Complex.init(0, 0));

        self.fft_input_buffer = try allocator.alloc(Complex, try getFFTLength(inputLen, kernelLen));
        @memset(self.fft_input_buffer, Complex.init(0, 0));

        self.fft_result_buffer = try allocator.alloc(Complex, try getFFTLength(inputLen, kernelLen));
        @memset(self.fft_result_buffer, Complex.init(0, 0));

        self.a_continuous = try allocator.alloc(Complex, numChannels);
        self.b_continuous = try allocator.alloc(Complex, numChannels);

        // 전달받은 가중치를 내부 메모리로 "복사"
        @memcpy(self.a_continuous, A);
        @memcpy(self.b_continuous, B);
        @memcpy(self.c_coeffs, C);

        try self.updateDiscretizedParams();
        try self.setupKernels();
        return self;
    }

    pub fn deinit(self: *S4Layer) void {
        const allocator = self.allocator;

        if (self.kernels.len > 0) {
            for (self.kernels) |k| {
                if (k.len > 0) {
                    allocator.free(k);
                }
            }
            allocator.free(self.kernels);
        }

        if (self.a_bars.len > 0) allocator.free(self.a_bars);
        if (self.b_bars.len > 0) allocator.free(self.b_bars);
        if (self.c_coeffs.len > 0) allocator.free(self.c_coeffs);
        if (self.states.len > 0) allocator.free(self.states);
        if (self.temp_buffer.len > 0) allocator.free(self.temp_buffer);
        if (self.output_buffer.len > 0) allocator.free(self.output_buffer);
        if (self.fft_input_buffer.len > 0) allocator.free(self.fft_input_buffer);
        if (self.fft_result_buffer.len > 0) allocator.free(self.fft_result_buffer);
        if (self.a_continuous.len > 0) allocator.free(self.a_continuous);
        if (self.b_continuous.len > 0) allocator.free(self.b_continuous);

        allocator.destroy(self);
    }

    pub fn updateDiscretizedParams(self: *S4Layer) !void {
        for (self.a_bars, 0..) |_, n| {
            // Define continuous-time dynamics (e.g., HiPPO-like initialization or oscillators)

            // Numerical discretization to obtain discrete transition matrices
            const discretized = try Complex.discretize(self.dt, self.a_continuous[n], self.b_continuous[n]);

            self.a_bars[n] = discretized.a_bar;
            self.b_bars[n] = discretized.b_bar;
        }
    }

    /// Parameter Initialization and Discretization:
    /// Maps continuous-time SSM parameters to discrete-time kernels.
    pub fn setupKernels(self: *S4Layer) !void {
        for (self.a_bars, 0..) |_, n| {
            // Compute the S4 Convolution Kernel: K_i = C_bar * A_bar^i * B_bar
            Complex.generateKernel(self.a_bars[n], self.b_bars[n], self.c_coeffs[n], self.kernels[n]);
        }
    }

    /// S4 Forward Pass (Convolutional View):
    /// Performs efficient signal transformation via parallel SIMD convolution.
    pub fn forward(self: *S4Layer, inputs: []const Complex) ![]Complex {
        std.debug.assert(inputs.len <= self.output_buffer.len);
        if (inputs.len > self.output_buffer.len) return error.InputLengthMisMatch;

        const len: usize = inputs.len;
        @memset(self.output_buffer[0..len], Complex.init(0.0, 0.0)); // Reset output for accumulation

        // Aggregate multi-channel filter responses
        for (self.c_coeffs, 0..) |_, n| {
            // Apply SIMD-accelerated time-domain convolution (equivalent to FFT-based conv for long-range dependencies)
            Complex.convolveSIMD(inputs, self.kernels[n], self.temp_buffer[0..len]);

            // Linearly combine channel responses into the final output projection
            Complex.addSIMD(self.temp_buffer[0..len], self.output_buffer[0..len], self.output_buffer[0..len]);
        }
        return self.output_buffer[0..len];
    }
};
