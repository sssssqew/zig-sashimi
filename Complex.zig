const std = @import("std");

pub const Complex = struct {
    re: f32,
    im: f32,

    pub const ComplexError = error{
        DivisionByZero,
    };

    pub fn init(re: f32, im: f32) Complex {
        return .{ .re = re, .im = im };
    }
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

            // calculate by simd
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

            // calculate by simd
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
    pub fn step(state: Complex, input: Complex, a: Complex, b: Complex) Complex {
        return a.mul(state).add(b.mul(input));
    }
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

    pub fn discretize(dt: f32, a: Complex, b: Complex) !struct { a_bar: Complex, b_bar: Complex } {
        std.debug.assert(dt > 0);
        if (dt <= 0) return error.InvalidTimeStep;

        const denom = a.scale(dt / 2.0 * -1).addReal(1);
        const da = try a.scale(dt / 2.0).addReal(1).div(denom);
        const db = try b.scale(dt).div(denom);
        return .{ .a_bar = da, .b_bar = db };
    }
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
                // calculate by simd
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

    const seq_len = 20;
    const n_channels = 8;
    const dt: f32 = 0.1;

    var a_weights = [n_channels]Complex{
        Complex.init(-1, 1.99),
        Complex.init(-1, 2.0),
        Complex.init(-1, 2.01),
        Complex.init(-1, 5.0),
        Complex.init(-1, 19.99),
        Complex.init(-1, 20.1),
        Complex.init(-1, 20),
        Complex.init(-1, 20.01),
    };
    var b_weights = [n_channels]Complex{
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
        Complex.init(1, 0),
    };
    var c_weights = [n_channels]Complex{
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
    };

    for (0..n_channels) |i| {
        const d = try Complex.discretize(dt, a_weights[i], b_weights[i]);
        a_weights[i] = d.a_bar;
        b_weights[i] = d.b_bar;
    }

    const inputs = try allocator.alloc(Complex, seq_len);
    defer allocator.free(inputs);

    for (inputs, 0..) |*in, i| {
        const t = @as(f32, @floatFromInt(i)) * dt;
        const signal = std.math.sin(t * 2.0);
        const noise = std.math.sin(t * 20.0);
        in.* = Complex.init(signal + noise, 0);
    }

    // const scan_results = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(scan_results);
    // Complex.scan(inputs, disc.a_bar, disc.b_bar, Complex.init(0, 0), scan_results);

    // for (scan_results) |*res| {
    //     res.* = res.*.mul(c);
    // }

    // const kernel = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(kernel);
    // Complex.generateKernel(disc.a_bar, disc.b_bar, Complex.init(1.0, 0.0), kernel);

    // const conv_results = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(conv_results);
    // Complex.convolveSIMD(inputs, kernel, conv_results);

    // for (0..seq_len) |i| {
    //     std.debug.print("Step {d}    | {d:.3} + {d:.3}i | {d:.3} + {d:.3}i\n", .{
    //         i,
    //         scan_results[i].re,
    //         scan_results[i].im,
    //         conv_results[i].re,
    //         conv_results[i].im,
    //     });
    // }

    // var c_train = Complex.init(0.1, 0.0);
    // const target = Complex.init(23.7, 6.9);

    // for (0..1000) |epoch| {
    //     const x = conv_results[4];
    //     const y = x.mul(c_train);

    //     // const loss = (y.re - target.re) * (y.re - target.re) + (y.im - target.im) * (y.im - target.im);

    //     const err = Complex.init(y.re - target.re, y.im - target.im);
    //     const grad_re = err.re * x.re + err.im * x.im;
    //     const grad_im = err.im * x.re - err.re * x.im;

    //     c_train.re -= grad_re * 0.001;
    //     c_train.im -= grad_im * 0.001;

    //     // if (epoch % 50 == 0) {
    //     //     std.debug.print("Epoch {d}: Loss = {d:.6}, Y={d:.3} + {d:.3}i\n", .{ epoch, loss, y.re, y.im });
    //     // }
    //     // if (epoch % 1 == 0) { // 모든 점을 다 찍어봅시다
    //     //     std.debug.print("{d:.6}, {d:.6}\n", .{ c_train.re, c_train.im });
    //     // }
    // }
    // const final_y = conv_results[4].mul(c_train);
    // std.debug.print("final estimated output: {d:.1} + {d:.1}i (Target was {d:.1} + {d:.1}i)\n", .{ final_y.re, final_y.im, target.re, target.im });

    const targets = try allocator.alloc(Complex, seq_len);
    defer allocator.free(targets);

    for (0..seq_len) |i| {
        const t = @as(f32, @floatFromInt(i)) * dt;
        targets[i] = Complex.init(std.math.sin(t * 2.0), 0);
    }

    for (0..10000) |epoch| {
        var total_loss: f32 = 0;
        var total_grad_a = [n_channels]Complex{
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
        };
        var total_grad_b = [n_channels]Complex{
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
        };
        var total_grad_c = [n_channels]Complex{
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
        };
        var states = [n_channels]Complex{
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
        };
        var prevStates = [n_channels]Complex{
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
            Complex.init(0, 0),
        };
        for (inputs, 0..) |u, i| {
            var output = Complex.init(0, 0);
            for (0..n_channels) |n| {
                prevStates[n] = states[n];
                const ax = states[n].mul(a_weights[n]);
                const bu = u.mul(b_weights[n]);
                states[n] = ax.add(bu);

                const y = states[n].mul(c_weights[n]);
                output = output.add(y);
            }
            const err = output.sub(targets[i]);
            const loss = ((output.re - targets[i].re) * (output.re - targets[i].re) + (output.im - targets[i].im) * (output.im - targets[i].im));
            total_loss += loss;

            // 에러누적
            for (0..n_channels) |n| {
                total_grad_a[n] = total_grad_a[n].add(err.mul(c_weights[n].conj()).mul(prevStates[n].conj()));
                total_grad_b[n] = total_grad_b[n].add(err.mul(c_weights[n].conj()).mul(u.conj()));
                total_grad_c[n] = total_grad_c[n].add(err.mul(states[n].conj()));
            }
        }
        // A,B,C update once
        for (0..n_channels) |n| {
            const inv_len = 1.0 / @as(f32, @floatFromInt(seq_len));
            a_weights[n] = a_weights[n].sub(total_grad_a[n].scale(0.01 * inv_len));
            b_weights[n] = b_weights[n].sub(total_grad_b[n].scale(0.05 * inv_len));
            c_weights[n] = c_weights[n].sub(total_grad_c[n].scale(0.5 * inv_len));

            // a_weights[n].re = @min(a_weights[n].re, -0.01); // 시스템 안정성 확보

            // const mag = std.math.sqrt(a_weights[n].re * a_weights[n].re + a_weights[n].im * a_weights[n].im);
            // if (mag > 0.999) {
            //     a_weights[n] = a_weights[n].scale(0.999 / mag);
            // }
            if (epoch % 500 == 0) {
                if (n == 3)
                    std.debug.print("Epoch {d} Channel {d}: Loss = {d:.6}, A = {d:.3} + {d:.3}i B = {d:.3} + {d:.3}i C = {d:.3} + {d:.3}i\n", .{ epoch, n, total_loss, a_weights[n].re, a_weights[n].im, b_weights[n].re, b_weights[n].im, c_weights[n].re, c_weights[n].im });
            }
        }
    }
    var test_states = [n_channels]Complex{
        Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0),
        Complex.init(0.0, 0.0),
    };
    std.debug.print("\n--- Final Verification ---\n", .{});
    for (inputs[0..10], 0..) |u, i| {
        var output = Complex.init(0, 0);
        for (0..n_channels) |n| {
            const next_state = test_states[n].mul(a_weights[n]).add(u.mul(b_weights[n]));
            const y = next_state.mul(c_weights[n]);
            output = output.add(y);
        }
        std.debug.print("Step {d}: Pred({d:.2} + {d:.2}i) vs Target({d:.2} + {d:.2}i)\n", .{ i, output.re, output.im, targets[i].re, targets[i].im });
    }
}
