const std = @import("std");

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
