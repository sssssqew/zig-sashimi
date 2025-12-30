const std = @import("std");
const Complex = @import("Complex.zig").Complex;

pub const S4Layer = struct {
    allocator: std.mem.Allocator,
    dt: f32,
    a_bars: []Complex,
    b_bars: []Complex,
    c_coeffs: []Complex,
    states: []Complex,
    kernels: [][]Complex,
    temp_buffer: []Complex, // forward 연산용 임시 작업대
    output_buffer: []Complex, // 최종 결과 저장용 버퍼

    pub fn init(allocator: std.mem.Allocator, numChannels: usize, kernelLen: usize, inputLen: usize, dt: f32) !*S4Layer {
        const self = try allocator.create(S4Layer);
        errdefer self.deinit();

        self.* = .{
            .allocator = allocator,
            .dt = dt,
            .a_bars = &[_]Complex{},
            .b_bars = &[_]Complex{},
            .c_coeffs = &[_]Complex{},
            .states = &[_]Complex{},
            .kernels = &[_][]Complex{},
            .temp_buffer = &[_]Complex{},
            .output_buffer = &[_]Complex{},
        };

        self.a_bars = try allocator.alloc(Complex, numChannels);
        self.b_bars = try allocator.alloc(Complex, numChannels);
        self.c_coeffs = try allocator.alloc(Complex, numChannels);
        self.states = try allocator.alloc(Complex, numChannels);
        @memset(self.states, Complex.init(0, 0));

        self.kernels = try allocator.alloc([]Complex, numChannels);
        @memset(self.kernels, &[_]Complex{});
        for (self.kernels) |*k| {
            k.* = try allocator.alloc(Complex, kernelLen);
        }
        self.temp_buffer = try allocator.alloc(Complex, inputLen);
        @memset(self.temp_buffer, Complex.init(0, 0));
        self.output_buffer = try allocator.alloc(Complex, inputLen);
        @memset(self.output_buffer, Complex.init(0, 0));

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

        allocator.destroy(self);
    }
    pub fn setupKernels(self: *S4Layer) !void {
        for (self.a_bars, 0..) |_, n| {
            const a_continuous = Complex.init(-0.5, @as(f32, @floatFromInt(n)) * std.math.pi);
            const b_continuous = Complex.init(1.0, 0.0);
            const discretized = try Complex.discretize(self.dt, a_continuous, b_continuous);

            self.c_coeffs[n] = Complex.init(1.0, 0.0);
            self.a_bars[n] = discretized.a_bar;
            self.b_bars[n] = discretized.b_bar;
            Complex.generateKernel(self.a_bars[n], self.b_bars[n], self.c_coeffs[n], self.kernels[n]);
        }
    }
    pub fn forward(self: *S4Layer, inputs: []const Complex) ![]Complex {
        std.debug.assert(inputs.len <= self.output_buffer.len);
        if (inputs.len > self.output_buffer.len) return error.InputLengthMisMatch;

        const len: usize = inputs.len;
        @memset(self.output_buffer[0..len], Complex.init(0.0, 0.0)); // 출력 버퍼 초기화 (누적 방지)

        for (self.c_coeffs, 0..) |_, n| {
            Complex.convolveSIMD(inputs, self.kernels[n], self.temp_buffer[0..len]);
            // for (temp) |*item| {
            //     item.* = item.*.mul(c);
            // }
            // Complex.mulScalarSIMD(c, self.temp_buffer[0..len], self.temp_buffer[0..len]); // c는 이미 generateKernel 함수에서 곱했으므로 필요없는 코드
            Complex.addSIMD(self.temp_buffer[0..len], self.output_buffer[0..len], self.output_buffer[0..len]);
        }
        return self.output_buffer[0..len];
    }
};
