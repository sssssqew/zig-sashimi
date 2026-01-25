// S4Trainer.zig
const std = @import("std");
const S4Layer = @import("S4.zig").S4Layer;
const Complex = @import("Complex.zig").Complex;

pub const TrainConfig = struct {
    lr_a: f32 = 0.01,
    lr_b: f32 = 0.05,
    lr_c: f32 = 0.5,
    epochs: usize = 10000,
    window_size: usize = 128,
};

/// 1. 메모리 효율을 위해 끊어서 학습
pub fn trainTruncatedBPTT(layer: *S4Layer, inputs: []const Complex, targets: []const Complex, config: TrainConfig) !void {
    // 여기에 지금 짜신 로직을 이식!
    // layer.a_bars, layer.b_bars 등을 업데이트하는 방식으로 구현
    const n_channels = layer.a_bars.len;
    const inv_len = 1.0 / @as(f32, @floatFromInt(config.window_size));

    // 1. 루프 시작 전 (또는 함수 초반) 메모리 할당
    const total_grad_a = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_b = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_c = try layer.allocator.alloc(Complex, n_channels);
    const prevStates = try layer.allocator.alloc(Complex, n_channels);
    defer {
        layer.allocator.free(total_grad_a);
        layer.allocator.free(total_grad_b);
        layer.allocator.free(total_grad_c);
        layer.allocator.free(prevStates);
    }

    // Training Loop: Optimizing via Backpropagation Through Time (BPTT)
    for (0..config.epochs) |epoch| {
        var start_idx: usize = 0;
        @memset(layer.states, Complex.init(0, 0));

        while (start_idx < inputs.len) : (start_idx += config.window_size) {
            var total_loss: f32 = 0;
            // 윈도우 시작할 때마다 0으로 초기화
            @memset(total_grad_a, Complex.init(0, 0));
            @memset(total_grad_b, Complex.init(0, 0));
            @memset(total_grad_c, Complex.init(0, 0));

            const end_idx = @min(start_idx + config.window_size, inputs.len);
            for (inputs[start_idx..end_idx], 0..) |u, i| {
                var output = Complex.init(0, 0);

                // Forward Pass: Calculating recurrent states across channels
                for (0..n_channels) |n| {
                    prevStates[n] = layer.states[n];
                    const ax = layer.states[n].mul(layer.a_bars[n]);
                    const bu = u.mul(layer.b_bars[n]);
                    layer.states[n] = ax.add(bu);

                    const y = layer.states[n].mul(layer.c_coeffs[n]);
                    output = output.add(y);
                }

                // Objective: Minimize L2 Distance (MSE) between Prediction and Target
                const err = output.sub(targets[start_idx + i]);
                const loss = err.re * err.re + err.im * err.im;
                total_loss += loss;

                // Backward Pass: Accumulate gradients using the Chain Rule (Sensitivity Analysis)
                for (0..n_channels) |n| {
                    total_grad_a[n] = total_grad_a[n].add(err.mul(layer.c_coeffs[n].conj()).mul(prevStates[n].conj()));
                    total_grad_b[n] = total_grad_b[n].add(err.mul(layer.c_coeffs[n].conj()).mul(u.conj()));
                    total_grad_c[n] = total_grad_c[n].add(err.mul(layer.states[n].conj()));
                }
            }

            // Global Parameter Update once per epoch
            for (0..n_channels) |n| {
                // Applying Gradient Descent
                layer.a_bars[n] = layer.a_bars[n].sub(total_grad_a[n].scale(config.lr_a * inv_len));
                layer.b_bars[n] = layer.b_bars[n].sub(total_grad_b[n].scale(config.lr_b * inv_len));
                layer.c_coeffs[n] = layer.c_coeffs[n].sub(total_grad_c[n].scale(config.lr_c * inv_len));

                // Spectral Radius Constraint: Enforce system stability (mag < 1.0)
                const mag = std.math.sqrt(layer.a_bars[n].re * layer.a_bars[n].re + layer.a_bars[n].im * layer.a_bars[n].im);
                if (mag > 0.999) {
                    layer.a_bars[n] = layer.a_bars[n].scale(0.999 / mag);
                }
                if (epoch % 100 == 0) {
                    std.debug.print("Epoch {d} Channel {d}: Loss = {d:.6}, A = {d:.3} + {d:.3}i B = {d:.3} + {d:.3}i C = {d:.3} + {d:.3}i\n", .{ epoch, n, total_loss, layer.a_bars[n].re, layer.a_bars[n].im, layer.b_bars[n].re, layer.b_bars[n].im, layer.c_coeffs[n].re, layer.c_coeffs[n].im });
                }
            }
        }
        // 커널 업데이트: 학습된 a, b, c를 다시 컨볼루션 커널로 변환 (추론을 위해)
        // try layer.setupKernels();
    }
}

// 2. 정석적인 전체 시퀀스 학습 (정답지)
// pub fn trainFullBPTT(layer: *S4.S4Layer, inputs: []const Complex, targets: []const Complex) !void {
//     // ...
// }

// /// 3. 대규모 가속 학습 (내일의 목표)
// pub fn trainConv(layer: *S4.S4Layer, inputs: []const Complex, targets: []const Complex) !void {
//     // FFT 기반 학습 로직
// }
