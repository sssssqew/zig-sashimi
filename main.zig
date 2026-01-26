const std = @import("std");
const S4Layer = @import("S4.zig").S4Layer;
const S4Trainer = @import("S4Trainer.zig");
const Complex = @import("Complex.zig").Complex;

fn generateSignal(freq: f32, t: f32) Complex {
    return std.math.sin(freq * t);
}
fn generateDataset(allocator: std.mem.Allocator, seqLen: usize, dt: f32) ![]Complex {
    const dataset = try allocator.alloc(Complex, seqLen);
    defer allocator.free(dataset);

    for (dataset, 0..) |*d, i| {
        const t = @as(f32, @floatFromInt(i)) * dt;
        const signal = generateSignal(2.0, t);
        const noise = generateSignal(20.0, t);
        d.* = Complex.init(signal + noise, 0);
    }
    return dataset;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    std.debug.print("Starting S4 Model test..\n", .{});

    // Configuration for Multi-channel State Space Model
    const seq_len: usize = 2000;
    const n_channels = 4;
    const dt: f32 = 0.2;
    const myConfig = S4Trainer.TrainConfig{
        .lr_a = 0.02,
        .lr_b = 0.05,
        .lr_c = 0.5,
        .epochs = 1000,
    };

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
    var c_weights = [n_channels]Complex{
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
        Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
    };

    const myLayer = try S4Layer.init(allocator, n_channels, 128, 128, dt, &a_weights, &b_weights, &c_weights);
    defer myLayer.deinit();

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
    for (inputs[0..100], 0..) |u, i| {
        var output = Complex.init(0, 0);
        for (0..n_channels) |n| {
            const next_state = test_states[n].mul(myLayer.a_bars[n]).add(u.mul(myLayer.b_bars[n]));
            test_states[n] = next_state; // Update state for inference
            const y = next_state.mul(myLayer.c_coeffs[n]);
            output = output.add(y);
        }
        std.debug.print("Step {d}: Pred({d:.2} + {d:.2}i) vs Target({d:.2} + {d:.2}i)\n", .{ i, output.re, output.im, targets[i].re, targets[i].im });
    }

    // 5. 학습 후 결과 확인 (Convolutional 모드 사용 가능)
    // const output = try myLayer.forward(inputs);
    // for (output) |o| {
    //     std.debug.print("학습 완료! 최종 결과 출력 중... {d:.2} + {d:.2}i\n", .{ o.re, o.im });
    // }
}
