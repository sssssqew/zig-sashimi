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
    const seq_len: usize = 1000;
    const n_channels = 4;
    const dt: f32 = 0.1;
    // const myConfig = S4Trainer.TrainConfig{
    //     .lr_a = 0.9,
    //     .lr_b = 0.9,
    //     .lr_c = 0.9,
    //     .epochs = 10000,
    // };

    // Weight Initialization (A, B, C parameters)
    var a_weights = [n_channels]Complex{
        Complex.init(-0.9, 0.5),
        Complex.init(-0.9, 2.0),
        Complex.init(-0.9, 10.0),
        Complex.init(-0.9, 30.0),
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

    const myLayer = try S4Layer.init(allocator, n_channels, seq_len, seq_len, dt, &a_weights, &b_weights, &c_weights);
    defer myLayer.deinit();

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
    // const targets = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(targets);

    // for (0..seq_len) |i| {
    //     const t = @as(f32, @floatFromInt(i)) * dt;
    //     targets[i] = Complex.init(std.math.sin(t * 2.0), 0);
    // }

    //
    // try S4Trainer.trainTruncatedBPTT(myLayer, inputs, targets, myConfig);
    // try S4Trainer.trainFullBPTT(myLayer, inputs, targets, myConfig);
    // const output = try S4Trainer.forwardConv(myLayer, inputs);
    // defer {
    //     for (output) |ch| allocator.free(ch);
    //     allocator.free(output);
    // }

    var total_loss: f32 = 0;

    // 1. FFT 기반 (CNN 모드) 결과 얻기
    const output_cnn = try S4Trainer.forwardConv(myLayer, inputs);
    defer {
        for (output_cnn) |ch| allocator.free(ch);
        allocator.free(output_cnn);
    }

    // 2. RNN 방식 (Step-by-step) 결과 계산 및 비교
    std.debug.print("\n--- Validation: RNN vs CNN ---\n", .{});
    var test_states = try allocator.alloc(Complex, n_channels);
    @memset(test_states, Complex.init(0, 0));
    defer allocator.free(test_states);

    for (0..seq_len) |i| {
        const u = inputs[i];
        var rnn_total_output = Complex.init(0, 0);

        for (0..n_channels) |n| {
            // RNN 방식 한 단계 업데이트
            const next_state = test_states[n].mul(myLayer.a_bars[n]).add(u.mul(myLayer.b_bars[n]));
            test_states[n] = next_state;
            const y = next_state.mul(myLayer.c_coeffs[n]);
            rnn_total_output = rnn_total_output.add(y);
        }

        // CNN 방식 결과 (모든 채널의 합을 구해야 함)
        var cnn_total_output = Complex.init(0, 0);
        for (0..n_channels) |n| {
            cnn_total_output = cnn_total_output.add(output_cnn[n][i]);
        }

        if (i < 15) { // 상위 5개만 샘플로 비교
            std.debug.print("Step {d}: RNN({d:.4}) vs CNN({d:.4})\n", .{ i, rnn_total_output.re, cnn_total_output.re });
        }
        total_loss += rnn_total_output.re - cnn_total_output.re;
    }

    std.debug.print("Total Loss: {d:.5}\n", .{total_loss});

    // Evaluation: Final Inference on the training sequence
    // var test_states = [_]Complex{Complex.init(0.0, 0.0)} ** n_channels;
    // std.debug.print("\n--- Final Verification ---\n", .{});
    // for (inputs, 0..) |u, i| {
    //     var output = Complex.init(0, 0);
    //     for (0..n_channels) |n| {
    //         const next_state = test_states[n].mul(myLayer.a_bars[n]).add(u.mul(myLayer.b_bars[n]));
    //         test_states[n] = next_state; // Update state for inference
    //         const y = next_state.mul(myLayer.c_coeffs[n]);
    //         output = output.add(y);
    //     }
    //     std.debug.print("Step {d}: Pred({d:.2} + {d:.2}i) vs Target({d:.2} + {d:.2}i)\n", .{ i, output.re, output.im, targets[i].re, targets[i].im });
    // }

    // var input = [_]Complex{
    //     Complex.init(1, 0),
    //     Complex.init(0, 0),
    //     Complex.init(0, 0),
    //     Complex.init(0, 0),
    // };
    // S4Trainer.fft(&input, input.len);
    // S4Trainer.ifft(&input, input.len);
    // for (input) |in| {
    //     std.debug.print("{d:.3} + {d:.3}i\n", .{ in.re, in.im });
    // }

    // 5. 학습 후 결과 확인 (Convolutional 모드 사용 가능)
    // const output = try myLayer.forward(inputs);
    // for (output) |o| {
    //     std.debug.print("학습 완료! 최종 결과 출력 중... {d:.2} + {d:.2}i\n", .{ o.re, o.im });
    // }

    // var a_weights = [_]Complex{
    //     Complex.init(-0.9, 0.5),
    // };
    // var b_weights = [_]Complex{
    //     Complex.init(1, 0),
    // };
    // var c_weights = [_]Complex{
    //     Complex.init(1.0 / @as(f32, @floatFromInt(1)), 0),
    // };

    // // 1. 가짜 S4Layer 설정 (테스트용)
    // var layer = try S4Layer.init(allocator, 1, 4, 4, dt, &a_weights, &b_weights, &c_weights); // 1채널, 시퀀스 길이 4
    // // defer myLayer.deinit();
    // defer layer.deinit();

    // // 2. 커널 강제 설정: [1, 1, 0, 0] (처음 두 칸만 활성화되는 필터)
    // layer.kernels[0][0] = Complex.init(1, 0);
    // layer.kernels[0][1] = Complex.init(1, 0);
    // layer.kernels[0][2] = Complex.init(0, 0);
    // layer.kernels[0][3] = Complex.init(0, 0);

    // // 3. 입력: [1, 0, 0, 0]
    // var input = [_]Complex{
    //     Complex.init(1, 0),
    //     Complex.init(0, 0),
    //     Complex.init(0, 0),
    //     Complex.init(0, 0),
    // };

    // // 4. 연산 수행
    // const result_channels = try S4Trainer.forwardConv(layer, &input);
    // defer {
    //     for (result_channels) |ch| allocator.free(ch);
    //     allocator.free(result_channels);
    // }

    // // 5. 결과 출력
    // std.debug.print("Convolution Result (Channel 0):\n", .{});
    // for (result_channels[0]) |val| {
    //     std.debug.print("{d:.1} + {d:.1}i\n", .{ val.re, val.im });
    // }
}
