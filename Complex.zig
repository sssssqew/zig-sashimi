const std = @import("std");
const S4Layer = @import("S4.zig").S4Layer;
const S4Trainer = @import("S4Trainer.zig");

/// Complex number structure for complex-valued neural network operations
pub const Complex = struct {
    re: f32,
    im: f32,

    pub const ComplexError = error{
        DivisionByZero,
    };

    pub fn init(re: f32, im: f32) Complex {
        return .{ .re = re, .im = im };
    }

    /// Complex conjugation for gradient calculation and Hermitian operations
    pub fn conj(self: Complex) Complex {
        return .{ .re = self.re, .im = self.im * -1 };
    }

    pub fn add(self: Complex, other: Complex) Complex {
        return .{ .re = self.re + other.re, .im = self.im + other.im };
    }

    pub fn sub(self: Complex, other: Complex) Complex {
        return .{ .re = self.re - other.re, .im = self.im - other.im };
    }

    pub fn mul(self: Complex, other: Complex) Complex {
        return .{ .re = self.re * other.re - self.im * other.im, .im = self.re * other.im + self.im * other.re };
    }

    /// SIMD-accelerated complex addition for high-performance sequence processing
    pub fn addSIMD(a: []const Complex, b: []const Complex, result: []Complex) void {
        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;
        const len = a.len;

        var i: usize = 0;
        while (i + vectorSize <= len) : (i += vectorSize) {
            var reA: @Vector(vectorSize, f32) = undefined;
            var imA: @Vector(vectorSize, f32) = undefined;
            var reB: @Vector(vectorSize, f32) = undefined;
            var imB: @Vector(vectorSize, f32) = undefined;

            inline for (0..vectorSize) |j| { // load
                reA[j] = a[i + j].re;
                imA[j] = a[i + j].im;
                reB[j] = b[i + j].re;
                imB[j] = b[i + j].im;
            }

            const sumRe: @Vector(vectorSize, f32) = reA + reB;
            const sumIm: @Vector(vectorSize, f32) = imA + imB;

            inline for (0..vectorSize) |j| { // store
                result[i + j].re = sumRe[j];
                result[i + j].im = sumIm[j];
            }
        }
        while (i < len) : (i += 1) {
            result[i] = a[i].add(b[i]);
        }
    }

    /// SIMD-accelerated complex multiplication
    pub fn mulSIMD(a: []const Complex, b: []const Complex, result: []Complex) void {
        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;
        const len = a.len;

        var i: usize = 0;
        while (i + vectorSize <= len) : (i += vectorSize) {
            var reA: @Vector(vectorSize, f32) = undefined;
            var imA: @Vector(vectorSize, f32) = undefined;
            var reB: @Vector(vectorSize, f32) = undefined;
            var imB: @Vector(vectorSize, f32) = undefined;

            inline for (0..vectorSize) |j| { // load
                reA[j] = a[i + j].re;
                imA[j] = a[i + j].im;
                reB[j] = b[i + j].re;
                imB[j] = b[i + j].im;
            }

            const mulRe: @Vector(vectorSize, f32) = (reA * reB) - (imA * imB);
            const mulIm: @Vector(vectorSize, f32) = (reA * imB) + (imA * reB);

            inline for (0..vectorSize) |j| { // store
                result[i + j].re = mulRe[j];
                result[i + j].im = mulIm[j];
            }
        }
        while (i < len) : (i += 1) {
            result[i] = a[i].mul(b[i]);
        }
    }

    /// Single time-step transition: x_t = A*x_{t-1} + B*u_t
    pub fn step(state: Complex, input: Complex, a: Complex, b: Complex) Complex {
        return a.mul(state).add(b.mul(input));
    }

    /// Linear Recurrent Scan: Parallelizable alternative to standard RNN loops
    pub fn scan(inputs: []const Complex, a: Complex, b: Complex, initialState: Complex, results: []Complex) void {
        std.debug.assert(inputs.len == results.len);

        var state: Complex = initialState;
        for (inputs, results) |input, *result| {
            state = step(state, input, a, b);
            result.* = state;
        }
    }

    pub fn addReal(self: Complex, s: f32) Complex {
        return .{ .re = self.re + s, .im = self.im };
    }

    pub fn scale(self: Complex, s: f32) Complex {
        return .{ .re = self.re * s, .im = self.im * s };
    }

    pub fn div(self: Complex, other: Complex) ComplexError!Complex {
        const denom = other.re * other.re + other.im * other.im;
        if (denom < 1e-12) return ComplexError.DivisionByZero;
        std.debug.assert(denom != 0);

        const re = (self.re * other.re + self.im * other.im) / denom;
        const im = (self.im * other.re - self.re * other.im) / denom;
        return .{ .re = re, .im = im };
    }

    /// Continuous-to-Discrete Bilinear Transformation (Tustin's Method)
    pub fn discretize(dt: f32, a: Complex, b: Complex) !struct { a_bar: Complex, b_bar: Complex } {
        std.debug.assert(dt > 0);
        if (dt <= 0) return error.InvalidTimeStep;

        const denom = a.scale(dt / 2.0 * -1).addReal(1);
        const da = try a.scale(dt / 2.0).addReal(1).div(denom);
        const db = try b.scale(dt).div(denom);
        return .{ .a_bar = da, .b_bar = db };
    }

    /// SSM Kernel Generation: Expands state-space parameters into a convolutional filter
    pub fn generateKernel(a_bar: Complex, b_bar: Complex, c: Complex, result: []Complex) void {
        if (result.len == 0) return;
        result[0] = c.mul(b_bar);
        var prev = result[0];

        for (result[1..]) |*k| {
            k.* = prev.mul(a_bar);
            prev = k.*;
        }
    }

    pub fn convolve(inputs: []const Complex, kernel: []const Complex, outputs: []Complex) void {
        std.debug.assert(inputs.len == outputs.len);
        if (inputs.len != outputs.len) return;

        for (outputs, 0..) |*o, i| {
            var sum = Complex.init(0, 0);
            const maxK = std.math.min(i + 1, kernel.len);

            for (kernel[0..maxK], 0..) |k, j| {
                sum = sum.add(inputs[i - j].mul(k));
            }
            o.* = sum;
        }
    }

    /// High-performance convolution implementation using SIMD
    pub fn convolveSIMD(inputs: []const Complex, kernel: []const Complex, outputs: []Complex) void {
        std.debug.assert(inputs.len == outputs.len);
        if (inputs.len != outputs.len) return;

        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;

        for (outputs, 0..) |*o, t| {
            var i: usize = 0;
            var sumReVector: @Vector(vectorSize, f32) = @splat(0.0);
            var sumImVector: @Vector(vectorSize, f32) = @splat(0.0);
            const len = @min(t + 1, kernel.len);

            while (i + vectorSize <= len) : (i += vectorSize) {
                var reA: @Vector(vectorSize, f32) = undefined;
                var imA: @Vector(vectorSize, f32) = undefined;
                var reB: @Vector(vectorSize, f32) = undefined;
                var imB: @Vector(vectorSize, f32) = undefined;

                inline for (0..vectorSize) |j| { // load
                    const idx = i + j;
                    reA[j] = inputs[t - idx].re;
                    imA[j] = inputs[t - idx].im;
                    reB[j] = kernel[idx].re;
                    imB[j] = kernel[idx].im;
                }
                const mulRe: @Vector(vectorSize, f32) = (reA * reB) - (imA * imB);
                const mulIm: @Vector(vectorSize, f32) = (reA * imB) + (imA * reB);
                sumReVector += mulRe;
                sumImVector += mulIm;
            }
            var totalSumRe: f32 = @reduce(.Add, sumReVector);
            var totalSumIm: f32 = @reduce(.Add, sumImVector);

            while (i < len) : (i += 1) {
                const out = inputs[t - i].mul(kernel[i]);
                totalSumRe += out.re;
                totalSumIm += out.im;
            }
            o.* = Complex.init(totalSumRe, totalSumIm);
        }
    }

    pub fn mulScalarSIMD(scalar: Complex, array: []const Complex, result: []Complex) void {
        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;
        const len = array.len;

        const reS: @Vector(vectorSize, f32) = @splat(scalar.re);
        const imS: @Vector(vectorSize, f32) = @splat(scalar.im);

        var i: usize = 0;
        while (i + vectorSize <= len) : (i += vectorSize) {
            var reA: @Vector(vectorSize, f32) = undefined;
            var imA: @Vector(vectorSize, f32) = undefined;

            inline for (0..vectorSize) |j| {
                reA[j] = array[i + j].re;
                imA[j] = array[i + j].im;
            }

            const mulRe: @Vector(vectorSize, f32) = (reA * reS) - (imA * imS);
            const mulIm: @Vector(vectorSize, f32) = (reA * imS) + (imA * reS);

            inline for (0..vectorSize) |j| { // store
                result[i + j].re = mulRe[j];
                result[i + j].im = mulIm[j];
            }
        }
        while (i < len) : (i += 1) {
            result[i] = array[i].mul(scalar);
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    std.debug.print("Starting S4 Model test..\n", .{});

    // Configuration for Multi-channel State Space Model
    const seq_len: usize = 2000;
    const n_channels = 4;
    const dt: f32 = 0.1;
    const myConfig = S4Trainer.TrainConfig{
        .epochs = 2000,
    };
    const myLayer = try S4Layer.init(allocator, n_channels, 128, 128, dt);
    defer myLayer.deinit();

    // Weight Initialization (A, B, C parameters)
    var a_weights = [n_channels]Complex{
        Complex.init(-1, 0.5),
        Complex.init(-1, 2.0),
        Complex.init(-1, 10.0),
        Complex.init(-1, 30.0),
    };
    var b_weights = [n_channels]Complex{
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
    };

    for (myLayer.c_coeffs) |*c| {
        c.* = Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0);
    }

    // Pre-processing: Continuous to Discrete mapping
    for (0..n_channels) |n| {
        const d = try Complex.discretize(dt, a_weights[n], b_weights[n]);
        myLayer.a_bars[n] = d.a_bar;
        myLayer.b_bars[n] = d.b_bar;
    }

    // Input Signal: Sinusoidal waves with high-frequency noise
    const inputs = try allocator.alloc(Complex, seq_len);
    defer allocator.free(inputs);

    for (inputs, 0..) |*in, i| {
        const t = @as(f32, @floatFromInt(i)) * dt;
        const signal = std.math.sin(t * 2.0);
        const noise = std.math.sin(t * 20.0);
        in.* = Complex.init(signal + noise, 0);
    }

    // Target Signal: Denoised sinusoidal wave for supervised learning
    const targets = try allocator.alloc(Complex, seq_len);
    defer allocator.free(targets);

    for (0..seq_len) |i| {
        const t = @as(f32, @floatFromInt(i)) * dt;
        targets[i] = Complex.init(std.math.sin(t * 2.0), 0);
    }

    //
    try S4Trainer.trainTruncatedBPTT(myLayer, inputs, targets, myConfig);

    // Evaluation: Final Inference on the training sequence
    var test_states = [_]Complex{Complex.init(0.0, 0.0)} ** n_channels;
    std.debug.print("\n--- Final Verification ---\n", .{});
    for (inputs[0..500], 0..) |u, i| {
        var output = Complex.init(0, 0);
        for (0..n_channels) |n| {
            const next_state = test_states[n].mul(myLayer.a_bars[n]).add(u.mul(myLayer.b_bars[n]));
            test_states[n] = next_state; // Update state for inference
            const y = next_state.mul(myLayer.c_coeffs[n]);
            output = output.add(y);
        }
        std.debug.print("Step {d}: Pred({d:.2} + {d:.2}i) vs Target({d:.2} + {d:.2}i)\n", .{ i, output.re, output.im, targets[i].re, targets[i].im });
    }
}
