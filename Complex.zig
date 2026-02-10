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
    pub fn square(self: Complex) Complex {
        return .{ .re = self.re * self.re - self.im * self.im, .im = 2 * self.re * self.im };
    }
    pub fn mulSIMDWithConj(a: []const Complex, b: []const Complex, result: []Complex) void {
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

            const mulRe: @Vector(vectorSize, f32) = (reA * reB) - (imA * -imB);
            const mulIm: @Vector(vectorSize, f32) = (reA * -imB) + (imA * reB);

            inline for (0..vectorSize) |j| { // store
                result[i + j].re = mulRe[j];
                result[i + j].im = mulIm[j];
            }
        }
        while (i < len) : (i += 1) {
            result[i] = a[i].mul(b[i].conj());
        }
    }
    pub fn conjSIMD(a: []const Complex) void {
        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;
        const len = a.len;

        var i: usize = 0;
        while (i + vectorSize <= len) : (i += vectorSize) {
            var reA: @Vector(vectorSize, f32) = undefined;
            var imA: @Vector(vectorSize, f32) = undefined;

            inline for (0..vectorSize) |j| { // load
                reA[j] = a[i + j].re;
                imA[j] = a[i + j].im;
            }
            imA *= -1;
            inline for (0..vectorSize) |j| { // store
                a[i + j].re = reA[j];
                a[i + j].im = imA[j];
            }
        }
        while (i < len) : (i += 1) {
            a[i] = a[i].conj();
        }
    }
    pub fn totalLossSIMD(a: []const Complex, b: []const Complex) f32 {
        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;
        const len = a.len;
        var total_sum: f32 = 0;

        var i: usize = 0;
        while (i + vectorSize <= len) : (i += vectorSize) {
            var reDiff: @Vector(vectorSize, f32) = undefined;
            var imDiff: @Vector(vectorSize, f32) = undefined;

            inline for (0..vectorSize) |j| { // load
                reDiff[j] = a[i + j].re - b[i + j].re;
                imDiff[j] = a[i + j].im - b[i + j].im;
            }

            const norm = reDiff * reDiff + imDiff * imDiff;
            total_sum += @reduce(.Add, norm);
        }
        while (i < len) : (i += 1) {
            const r = a[i].re - b[i].re;
            const im = a[i].im - b[i].im;
            total_sum += r * r + im * im;
        }
        return total_sum;
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
    /// SIMD-accelerated complex subtraction for high-performance sequence processing
    pub fn subSIMD(a: []const Complex, b: []const Complex, result: []Complex) void {
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

            const sumRe: @Vector(vectorSize, f32) = reA - reB;
            const sumIm: @Vector(vectorSize, f32) = imA - imB;

            inline for (0..vectorSize) |j| { // store
                result[i + j].re = sumRe[j];
                result[i + j].im = sumIm[j];
            }
        }
        while (i < len) : (i += 1) {
            result[i] = a[i].sub(b[i]);
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
    pub fn generateKernelNormal(a_bar: Complex, b_bar: Complex, c: Complex, result: []Complex) !void {
        std.debug.assert(result.len != 0);
        std.debug.assert(a_bar.mag() < 1.0);
        if (result.len == 0) return;
        if (a_bar.mag() >= 1.0) return error.UnstableSystem;

        result[0] = c.mul(b_bar);
        var prev = result[0];
        const threshold = 1e-9;

        for (result[1..], 0..) |*k, i| {
            k.* = prev.mul(a_bar);
            prev = k.*;

            if (k.mag() < threshold) { // A값이 0에 근접할때 조기종료
                @memset(result[i + 1 ..], Complex.init(0, 0));
                std.debug.print("--- Early Exit at t = {d} --- (M = {e})\n", .{ i, k.mag() });
                break;
            }
        }
    }

    fn mag(self: Complex) f32 {
        return std.math.sqrt(self.re * self.re + self.im * self.im);
    }
    fn phase(self: Complex) f32 {
        return std.math.atan2(self.im, self.re);
    }
    pub fn ln(self: Complex) Complex {
        return .{ .re = std.math.log(f32, std.math.e, self.mag()), .im = self.phase() };
    }

    pub fn generateKernel(a_bar: Complex, b_bar: Complex, c: Complex, result: []Complex) !void {
        const seq_len = result.len;

        // 1024를 기준으로 전략 선택
        if (seq_len <= 1024) {
            // 짧은 시퀀스: 오차가 적고 오버헤드에 민감하므로 순차 곱셈!
            return try generateKernelSequential(a_bar, b_bar, c, result);
        } else {
            // 긴 시퀀스: 누적 오차 방지 및 하드웨어 가속을 위해 SIMD 로그 방식!
            // (만약 SIMD 최적화가 안 되어 있다면 NormalLog라도 써야 함)
            return try generateKernelWithLogAndSIMD(a_bar, b_bar, c, result);
        }
    }

    pub fn generateKernelWithLog(a_bar: Complex, b_bar: Complex, c: Complex, result: []Complex) !void {
        std.debug.assert(result.len != 0);
        std.debug.assert(a_bar.mag() < 1.0);
        if (result.len == 0) return;
        if (a_bar.mag() >= 1.0) return error.UnstableSystem;

        const logA = a_bar.ln();
        const logB = b_bar.ln();
        const logC = c.ln();

        const fixedRe = logC.re + logB.re;
        const fixedIm = logC.im + logB.im;

        const threshold = 1e-9;
        for (result, 0..) |*r, i| {
            const i_f = @as(f32, @floatFromInt(i));
            const R = fixedRe + i_f * logA.re;
            const I = fixedIm + i_f * logA.im;
            const M = std.math.exp(R);

            if (M < threshold) { // A값이 0에 근접할때 조기종료
                @memset(result[i..], Complex.init(0, 0));
                // std.debug.print("--- Early Exit at t = {d} --- (M = {e})\n", .{ i, M });
                break;
            }

            r.* = Complex.init(M * std.math.cos(I), M * std.math.sin(I));
        }
    }

    pub fn generateKernelWithLogAndSIMD(a_bar: Complex, b_bar: Complex, c: Complex, result: []Complex) !void {
        std.debug.assert(result.len != 0);
        std.debug.assert(a_bar.mag() < 1.0);
        if (result.len == 0) return;
        if (a_bar.mag() >= 1.0) return error.UnstableSystem;

        const logA = a_bar.ln();
        const logB = b_bar.ln();
        const logC = c.ln();

        const fixedRe = logC.re + logB.re;
        const fixedIm = logC.im + logB.im;

        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;
        const logAreV: @Vector(vectorSize, f32) = @splat(logA.re); // broadcasting
        const logAimV: @Vector(vectorSize, f32) = @splat(logA.im);
        const fixedReV: @Vector(vectorSize, f32) = @splat(fixedRe);
        const fixedImV: @Vector(vectorSize, f32) = @splat(fixedIm);
        var IfV: @Vector(vectorSize, f32) = std.simd.iota(f32, vectorSize);
        const stepV: @Vector(vectorSize, f32) = @splat(@as(f32, @floatFromInt(vectorSize)));

        var RV: @Vector(vectorSize, f32) = undefined;
        var IV: @Vector(vectorSize, f32) = undefined;
        var MV: @Vector(vectorSize, f32) = undefined;

        var i: usize = 0;
        const len: usize = result.len;
        const threshold = 1e-9;

        while (i + vectorSize <= len) : (i += vectorSize) {
            RV = fixedReV + IfV * logAreV;
            IV = fixedImV + IfV * logAimV;

            inline for (0..vectorSize) |j| {
                MV[j] = std.math.exp(RV[j]);
                result[i + j] = Complex.init(MV[j] * std.math.cos(IV[j]), MV[j] * std.math.sin(IV[j]));
            }
            if (@reduce(.Max, MV) < threshold) {
                @memset(result[i + vectorSize ..], Complex.init(0, 0));
                // std.debug.print("--- Early Exit at t = {d} --- (M = {e})\n", .{ i, @reduce(.Max, MV) });
                return;
            }

            IfV = IfV + stepV;
        }
        while (i < len) : (i += 1) {
            const i_f = @as(f32, @floatFromInt(i));
            const R = fixedRe + i_f * logA.re;
            const I = fixedIm + i_f * logA.im;
            const M = std.math.exp(R);

            if (M < threshold) { // A값이 0에 근접할때 조기종료
                @memset(result[i..], Complex.init(0, 0));
                // std.debug.print("--- Early Exit at t = {d} --- (M = {e})\n", .{ i, M });
                return;
            }
            result[i] = Complex.init(M * std.math.cos(I), M * std.math.sin(I));
        }
    }

    pub fn generateKernelSequential(a_bar: Complex, b_bar: Complex, c: Complex, result: []Complex) !void {
        std.debug.assert(result.len != 0);
        if (a_bar.mag() >= 1.0) return error.UnstableSystem;

        const logA = a_bar.ln();
        const logB = b_bar.ln();
        const logC = c.ln();

        const fixedRe = logC.re + logB.re;
        const fixedIm = logC.im + logB.im;

        // --- 루프 밖에서 단 한 번만 수행하는 무거운 연산 ---
        // 1. 초기값 (t=0)
        const M_start = std.math.exp(fixedRe);
        const W_start = Complex.init(std.math.cos(fixedIm), std.math.sin(fixedIm));

        // 2. 매 스텝 곱해줄 변화량 (Step)
        const M_step = std.math.exp(logA.re);
        const W_step = Complex.init(std.math.cos(logA.im), std.math.sin(logA.im));

        var current_M = M_start;
        var current_W = W_start;

        const threshold = 1e-9;

        for (result, 0..) |*r, i| {
            // 루프 안에는 exp, cos, sin이 하나도 없습니다!
            // 오직 실수 곱셈과 복소수 곱셈뿐입니다.
            r.* = Complex.init(current_M * current_W.re, current_M * current_W.im);

            // 다음 단계 업데이트
            current_M *= M_step; // 크기 업데이트 (지수 법칙)
            current_W = current_W.mul(W_step); // 각도 업데이트 (회전)

            // 조기 종료 체크
            if (current_M < threshold) {
                @memset(result[i + 1 ..], Complex.init(0, 0));
                // std.debug.print("--- Early Exit (Sequential) at t = {d} ---\n", .{i});
                break;
            }
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

    // 복수소 연산 병렬화 (SIMD)
    fn Vector(comptime n: usize) type {
        return @Vector(n, f32);
    }

    fn ComplexVector(comptime n: usize) type {
        return struct {
            reV: Vector(n),
            imV: Vector(n),
        };
    }

    // inline : 함수 코드를 컴파일할때 호출부분에 넣어서 함수 호출 오버헤드 감소
    inline fn vLoad(ptr: [*]const Complex, comptime n: usize) Vector(n * 2) {
        return @bitCast(ptr[0..n].*);
    }
    inline fn vStore(ptr: [*]Complex, comptime n: usize, cv: Vector(n * 2)) void {
        ptr[0..n].* = @bitCast(cv);
    }
    inline fn vConj(comptime n: usize, cv: ComplexVector(n)) ComplexVector(n) {
        return .{ .reV = cv.reV, .imV = -cv.imV };
    }
    inline fn vScale(comptime n: usize, cv: ComplexVector(n), s: f32) ComplexVector(n) {
        const sV: Vector(n) = @splat(s);
        return .{ .reV = cv.reV * sV, .imV = cv.imV * sV };
    }
    inline fn vAdd(comptime n: usize, aV: ComplexVector(n), bV: ComplexVector(n)) ComplexVector(n) {
        return .{ .reV = aV.reV + bV.reV, .imV = aV.imV + bV.imV };
    }
    inline fn vSub(comptime n: usize, aV: ComplexVector(n), bV: ComplexVector(n)) ComplexVector(n) {
        return .{ .reV = aV.reV - bV.reV, .imV = aV.imV - bV.imV };
    }
    inline fn vMul(comptime n: usize, aV: ComplexVector(n), bV: ComplexVector(n)) ComplexVector(n) {
        const resRe = (aV.reV * bV.reV) - (aV.imV * bV.imV);
        const resIm = (aV.reV * bV.imV) + (aV.imV * bV.reV);
        return .{ .reV = resRe, .imV = resIm };
    }
    inline fn vDiv(comptime n: usize, aV: ComplexVector(n), bV: ComplexVector(n)) ComplexError!ComplexVector(n) {
        const denom = bV.reV * bV.reV + bV.imV * bV.imV;
        if (@reduce(.Or, denom < @as(Vector(n), @splat(1e-12)))) return ComplexError.DivisionByZero;
        std.debug.assert(denom != 0);

        const reV = (aV.reV * bV.reV + aV.imV * bV.imV) / denom;
        const imV = (aV.imV * bV.reV - aV.reV * bV.imV) / denom;
        return .{ .reV = reV, .imV = imV };
    }
    inline fn vOp(comptime n: usize, aV: ComplexVector(n), bV: ComplexVector(n), comptime mode: OpMode) !ComplexVector(n) {
        return switch (mode) {
            .add => vAdd(n, aV, bV),
            .sub => vSub(n, aV, bV),
            .mul => vMul(n, aV, bV),
            .div => try vDiv(n, aV, bV),
        };
    }
    inline fn cOp(a: Complex, b: Complex, comptime mode: OpMode) !Complex {
        return switch (mode) {
            .add => a.add(b),
            .sub => a.sub(b),
            .mul => a.mul(b),
            .div => try a.div(b),
        };
    }
    inline fn prepareComplexVector(
        comptime n: usize,
        ptr: [*]const Complex,
        comptime reMask: [n]i32,
        comptime imMask: [n]i32,
        comptime option: arrOption,
    ) ComplexVector(n) {
        const raw: Vector(n * 2) = vLoad(ptr, n);
        var cv: ComplexVector(n) = shuffleComplex(n, raw, reMask, imMask);
        if (option.conj) cv = vConj(n, cv);
        if (option.scale != 1.0) cv = vScale(n, cv, option.scale);
        return cv;
    }
    inline fn prepareComplex(c: Complex, comptime option: arrOption) Complex {
        var refined = c;
        if (option.conj) refined = refined.conj();
        if (option.scale != 1.0) refined = refined.scale(option.scale);
        return refined;
    }
    // comptime: 컴파일 시간에 결정되어야 할 값 (타입생성, 배열크기 등)
    fn makeSuffleMask(comptime n: usize, comptime offset: usize) [n]i32 {
        var mask: [n]i32 = undefined;
        for (&mask, 0..) |*m, i| {
            m.* = @intCast(i * 2 + offset);
        }
        return mask;
    }
    fn makeStoreMask(comptime n: usize) [n * 2]i32 {
        var mask: [n * 2]i32 = undefined;
        for (0..n) |i| {
            mask[i * 2] = i;
            mask[i * 2 + 1] = i + n;
        }
        return mask;
    }
    inline fn shuffleComplex(comptime n: usize, cv: @Vector(n * 2, f32), comptime reMask: [n]i32, comptime imMask: [n]i32) ComplexVector(n) {
        const reV = @shuffle(f32, cv, undefined, reMask);
        const imV = @shuffle(f32, cv, undefined, imMask);
        return .{ .reV = reV, .imV = imV };
    }
    inline fn restoreComplex(comptime n: usize, cv: ComplexVector(n), comptime mask: [n * 2]i32) Vector(n * 2) {
        return @shuffle(f32, cv.reV, cv.imV, mask);
    }
    const OpMode = enum { mul, add, sub, div };
    const arrOption = struct {
        conj: bool = false,
        scale: f32 = 1.0,
    };
    const convOptions = struct {
        mode: OpMode = .mul,
        a: arrOption = .{},
        b: arrOption = .{},
        out: arrOption = .{},
        acc_index: bool = false,
        acc_total: bool = false,
    };

    pub fn generalComplexOpSIMD(
        a: []const Complex,
        b: ?[]const Complex,
        result: []Complex,
        comptime opt: convOptions,
    ) !Complex {
        const vectorSize: usize = std.simd.suggestVectorLength(f32) orelse 4;
        const numOfComplex = vectorSize / 2;
        const len = a.len;

        const reMask: [numOfComplex]i32 = comptime makeSuffleMask(numOfComplex, 0);
        const imMask: [numOfComplex]i32 = comptime makeSuffleMask(numOfComplex, 1);
        const storeMask: [vectorSize]i32 = comptime makeStoreMask(numOfComplex);

        var i: usize = 0;
        var totalReV: Vector(numOfComplex) = @splat(0.0);
        var totalImV: Vector(numOfComplex) = @splat(0.0);
        while (i + numOfComplex <= len) : (i += numOfComplex) {
            var resV: ComplexVector(numOfComplex) = prepareComplexVector(numOfComplex, a.ptr + i, reMask, imMask, opt.a);

            if (b) |bSafe| {
                const bV = prepareComplexVector(numOfComplex, bSafe.ptr + i, reMask, imMask, opt.b);
                resV = try vOp(numOfComplex, resV, bV);
            }

            if (opt.out.conj) resV = vConj(numOfComplex, resV);
            if (opt.out.scale != 1.0) resV = vScale(numOfComplex, resV, opt.out.scale);

            const combined: Vector(vectorSize) = restoreComplex(numOfComplex, resV, storeMask);

            if (opt.acc_index) {
                const currentVal: Vector(vectorSize) = vLoad(result.ptr + i, numOfComplex); // 1. 기존 메모리 값을 로드 (복소수 n개만큼)
                vStore(result.ptr + i, numOfComplex, combined + currentVal); // 2. 벡터 더하기 (한 번에 8개의 f32가 더해집니다)
            } else {
                vStore(result.ptr + i, numOfComplex, combined);
            }

            if (opt.acc_total) {
                totalReV += resV.reV;
                totalImV += resV.imV;
            }
        }
        var total: Complex = Complex.init(0.0, 0.0);
        while (i < len) : (i += 1) {
            var res = prepareComplex(a[i], opt.a);

            if (b) |bSafe| {
                const refinedB = prepareComplex(bSafe[i], opt.b);
                res = try res.cOp(refinedB);
            }
            if (opt.out.conj) res = res.conj();
            if (opt.out.scale != 1.0) res = res.scale(opt.out.scale);

            if (opt.acc_index) {
                result[i] += res;
            } else {
                result[i] = res;
            }
            if (opt.acc_total) {
                total = total.add(res);
            }
        }
        if (opt.acc_total) {
            return .{ .re = @reduce(.Add, totalReV) + total.re, .im = @reduce(.Add, totalImV) + total.im };
        }
        return .{ .re = 0.0, .im = 0.0 };
    }
};
