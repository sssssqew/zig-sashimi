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
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // const allocator = gpa.allocator();
    // defer _ = gpa.deinit();

    // std.debug.print("Starting S4 Model test..\n", .{});

    // // Configuration for Multi-channel State Space Model
    // const seq_len: usize = 500;
    // const n_channels = 4;
    // const dt: f32 = 0.001;
    // const myConfig = S4Trainer.TrainConfig{
    //     .lr_a = 0.007, // 조금 더 자유롭게
    //     .lr_b = 0.01, // 입력 통로 확장
    //     .lr_c = 0.01, // 정밀한 수렴을 위해 기존 0.02에서 절반으로 축소
    //     .epochs = 5000,
    // };

    // var a_weights = try allocator.alloc(Complex, n_channels);
    // defer allocator.free(a_weights); // 이 줄을 추가하세요!

    // for (0..n_channels) |n| {
    //     const n_f = @as(f32, @floatFromInt(n));
    //     const n_chan_f = @as(f32, @floatFromInt(n_channels));

    //     // 1. 실수부는 -0.5로 고정 (시스템의 안정성)
    //     const re = -0.5;

    //     // 2. 허수부(주파수)를 HiPPO 방식으로 배치
    //     // 낮은 주파수부터 높은 주파수까지 로그 스케일로 정교하게 매핑
    //     const im = std.math.pi * std.math.pow(f32, 10.0, (n_f / (n_chan_f - 1)) * 2.0 - 1.0);
    //     // 위 수식은 대략 0.3 ~ 300 사이의 주파수를 골고루 뿌려줍니다.

    //     a_weights[n] = Complex.init(re, im);
    // }

    // // Weight Initialization (A, B, C parameters)
    // // var a_weights = [n_channels]Complex{
    // //     Complex.init(-1.0, 1.0), // 저주파 담당
    // //     Complex.init(-10.0, 10.0), // 중주파 담당
    // //     Complex.init(-100.0, 100.0), // 고주파 담당
    // //     Complex.init(-1000.0, 1000.0), // 초고주파 담당
    // // };

    // var b_weights = [n_channels]Complex{
    //     Complex.init(1, 0),
    //     Complex.init(1, 0),
    //     Complex.init(1, 0),
    //     Complex.init(1, 0),
    // };
    // var c_weights = [n_channels]Complex{
    //     Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
    //     Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
    //     Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
    //     Complex.init(1.0 / @as(f32, @floatFromInt(n_channels)), 0),
    // };

    // const myLayer = try S4Layer.init(allocator, n_channels, seq_len, seq_len, dt, a_weights, &b_weights, &c_weights);
    // defer myLayer.deinit();

    // // Input Signal: Sinusoidal waves with high-frequency noise
    // const inputs = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(inputs);

    // for (inputs, 0..) |*in, i| {
    //     const t = @as(f32, @floatFromInt(i)) * dt;
    //     const signal = std.math.sin(t * 2.0);
    //     const noise = std.math.sin(t * 20.0);
    //     in.* = Complex.init(signal + noise, 0);
    // }

    // // Target Signal: Denoised sinusoidal wave for supervised learning
    // const targets = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(targets);

    // for (0..seq_len) |i| {
    //     const t = @as(f32, @floatFromInt(i)) * dt;
    //     targets[i] = Complex.init(std.math.sin(t * 2.0), 0);
    // }

    // // 1. 학습 전 상태 확인
    // std.debug.print("\n--- Before Training ---\n", .{});
    // try S4Trainer.predict(myLayer, inputs);
    // var initial_loss: f32 = 0;
    // for (0..seq_len) |i| {
    //     const diff = myLayer.output_buffer[i].re - targets[i].re;
    //     initial_loss += diff * diff; // MSE 방식
    // }
    // std.debug.print("Initial MSE Loss: {d:.6}\n", .{initial_loss / @as(f32, @floatFromInt(seq_len))});

    // // 2. 본격적인 학습 시작
    // std.debug.print("\n--- Starting Training ---\n", .{});
    // // try S4Trainer.trainTruncatedBPTT(myLayer, inputs, targets, myConfig);
    // // try S4Trainer.trainFullBPTT(myLayer, inputs, targets, myConfig);
    // try S4Trainer.trainConv(myLayer, inputs, targets, myConfig);

    // // 3. 학습 후 결과 검증 (RNN vs CNN 일치 여부 포함)
    // std.debug.print("\n--- After Training: Validation ---\n", .{});
    // try S4Trainer.predict(myLayer, inputs);

    // var total_loss: f32 = 0;
    // var test_states = try allocator.alloc(Complex, n_channels);
    // @memset(test_states, Complex.init(0, 0));
    // defer allocator.free(test_states);

    // for (0..seq_len) |i| {
    //     const u = inputs[i];
    //     var rnn_total_output = Complex.init(0, 0);

    //     for (0..n_channels) |n| {
    //         // RNN 방식 한 단계 업데이트
    //         const next_state = test_states[n].mul(myLayer.a_bars[n]).add(u.mul(myLayer.b_bars[n]));
    //         test_states[n] = next_state;
    //         const y = next_state.mul(myLayer.c_coeffs[n]);
    //         rnn_total_output = rnn_total_output.add(y);
    //     }
    //     const r: f32 = myLayer.output_buffer[i].re - targets[i].re;
    //     const im: f32 = myLayer.output_buffer[i].im - targets[i].im;
    //     total_loss += r * r + im * im;

    //     if (i < 15) { // 상위 5개만 샘플로 비교
    //         std.debug.print("Step {d}: RNN({d:.4}+{d:.4}i) vs CNN({d:.4}+{d:.4}i)\n", .{ i, rnn_total_output.re, rnn_total_output.im, myLayer.output_buffer[i].re, myLayer.output_buffer[i].im });
    //     }
    // }

    // std.debug.print("\nFinal MSE Loss: {d:.6}\n", .{total_loss / @as(f32, @floatFromInt(seq_len))});
    // std.debug.print("Training Improvement: {d:.2}%\n", .{(initial_loss - total_loss) / initial_loss * 100});

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

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // const allocator = std.heap.page_allocator;
    // const seq_len = 300000; // 테스트용 길이는 짧게 시작

    // // 1. 테스트용 파라미터 설정 (3 + 2i 같은 값들)
    // const a_bar = Complex.init(0.999, 0.0001); // 안정적인 시스템을 위해 크기가 1보다 작게
    // const b_bar = Complex.init(1.0, 0.5);
    // const c = Complex.init(0.5, -0.2);

    // // 2. 결과 저장용 메모리 할당
    // const res_linear = try allocator.alloc(Complex, seq_len);
    // const res_log = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(res_linear);
    // defer allocator.free(res_log);

    // var timer = try std.time.Timer.start();

    // // 3. 두 방식 실행
    // timer.reset();
    // try Complex.generateKernel(a_bar, b_bar, c, res_linear); // 기존 방식
    // const linear_time = timer.read(); // 나노초(ns) 단위
    // timer.reset();
    // try Complex.generateKernelWithLog(a_bar, b_bar, c, res_log); // 로그 방식
    // const log_time = timer.read();

    // // 4. 결과 비교 출력
    // std.debug.print("{s:>5} | {s:>14} | {s:>14} | {s:>10}\n", .{ "t", "Linear", "Log", "Error" });
    // std.debug.print("--------------------------------------------------------------------------------\n", .{});

    // var total_loss: f32 = 0;
    // for (0..seq_len) |i| {
    //     const l = res_linear[i];
    //     const log_v = res_log[i];

    //     // 오차 계산 (실수부와 허수부 차이의 합)
    //     const err_re = @abs(l.re - log_v.re);
    //     const err_im = @abs(l.im - log_v.im);
    //     total_loss += err_re * err_re + err_im * err_im;

    //     std.debug.print("{d:>5} | {d:.15}+{d:.15}i | {d:.15}+{d:.15}i | {d:.15}\n", .{
    //         i, l.re, l.im, log_v.re, log_v.im, err_re + err_im,
    //     });
    // }
    // std.debug.print("total loss: {d:.10}\n", .{total_loss});

    // // 결과 출력
    // std.debug.print("\n[성능 비교 결과]\n", .{});
    // std.debug.print("리니어 방식: {d:>10} ns ({d:.3} ms)\n", .{ linear_time, @as(f32, @floatFromInt(linear_time)) / 1_000_000.0 });
    // std.debug.print("로그 방식  : {d:>10} ns ({d:.3} ms)\n", .{ log_time, @as(f32, @floatFromInt(log_time)) / 1_000_000.0 });

    // const ratio = @as(f32, @floatFromInt(log_time)) / @as(f32, @floatFromInt(linear_time));
    // std.debug.print("상대적 속도: 로그 방식이 리니어보다 {d:.2}배 더 걸림\n", .{ratio});

    // 커널생성 SIMD 여부에 따른 성능 테스트
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // const allocator = gpa.allocator();
    // defer _ = gpa.deinit();

    // // 1. 테스트 데이터 준비 (시퀀스 길이를 좀 길게 잡아야 차이가 확 보입니다)
    // const seq_len = 1000;
    // const result_base = try allocator.alloc(Complex, seq_len);
    // const result_normal = try allocator.alloc(Complex, seq_len);
    // const result_simd = try allocator.alloc(Complex, seq_len);
    // const result_seq = try allocator.alloc(Complex, seq_len);
    // const result_optimized = try allocator.alloc(Complex, seq_len);
    // defer allocator.free(result_base);
    // defer allocator.free(result_normal);
    // defer allocator.free(result_simd);
    // defer allocator.free(result_seq);
    // defer allocator.free(result_optimized);

    // // 테스트용 파라미터 (S4 논문에 나올 법한 값들)
    // const a_bar = Complex.init(0.99999, 0.0001);
    // const b_bar = Complex.init(0.5, -0.2);
    // const c = Complex.init(0.3, 0.4);

    // var timer = try std.time.Timer.start();

    // // --- Base Loop 측정 ---
    // const start_base = timer.read();
    // try Complex.generateKernelNormal(a_bar, b_bar, c, result_base);
    // const end_base = timer.read();
    // const duration_base = end_base - start_base;

    // // --- Normal Log Loop 측정 ---
    // const start_normal = timer.read();
    // try Complex.generateKernelWithLog(a_bar, b_bar, c, result_normal);
    // const end_normal = timer.read();
    // const duration_normal = end_normal - start_normal;

    // // --- SIMD Log Loop 측정 ---
    // const start_simd = timer.read();
    // try Complex.generateKernelWithLogAndSIMD(a_bar, b_bar, c, result_simd);
    // const end_simd = timer.read();
    // const duration_simd = end_simd - start_simd;

    // // --- Seqencial Loop 측정 ---
    // const start_normal_seq = timer.read();
    // try Complex.generateKernelSequential(a_bar, b_bar, c, result_seq);
    // const end_normal_seq = timer.read();
    // const duration_normal_seq = end_normal_seq - start_normal_seq;

    // // --- Optimized SIMD 측정 ---
    // const start_normal_optimized = timer.read();
    // try Complex.generateKernel(a_bar, b_bar, c, result_optimized);
    // const end_normal_optimized = timer.read();
    // const duration_normal_optimized = end_normal_optimized - start_normal_optimized;

    // // 2. 결과 출력 (좌측 정렬 너비 30, 우측 정렬 너비 12로 통일) /
    // std.debug.print("\n=== Benchmark Results (Seq Len: {d}) ===\n", .{seq_len});
    // std.debug.print("{s:<30}: {d:>12} ns\n", .{ "Base Loop", duration_base });
    // std.debug.print("{s:<30}: {d:>12} ns\n", .{ "Normal Log Loop", duration_normal });
    // std.debug.print("{s:<30}: {d:>12} ns\n", .{ "SIMD Log Loop", duration_simd });
    // std.debug.print("{s:<30}: {d:>12} ns\n", .{ "Sequential Loop", duration_normal_seq });
    // std.debug.print("{s:<30}: {d:>12} ns\n", .{ "Hybrid Dispatcher", duration_normal_optimized });

    // std.debug.print("\n--- Speedup Analysis (vs SIMD) ---\n", .{});
    // const speedup_base = @as(f64, @floatFromInt(duration_base)) / @as(f64, @floatFromInt(duration_simd));
    // const speedup_normal = @as(f64, @floatFromInt(duration_normal)) / @as(f64, @floatFromInt(duration_simd));
    // const speedup_seq = @as(f64, @floatFromInt(duration_normal_seq)) / @as(f64, @floatFromInt(duration_simd));
    // const speedup_optimized = @as(f64, @floatFromInt(duration_normal_optimized)) / @as(f64, @floatFromInt(duration_simd));

    // std.debug.print("{s:<30}: {d:>12.2}x\n", .{ "Speedup (Base/SIMD)", speedup_base });
    // std.debug.print("{s:<30}: {d:>12.2}x\n", .{ "Speedup (Normal/SIMD)", speedup_normal });
    // std.debug.print("{s:<30}: {d:>12.2}x\n", .{ "Speedup (Sequential/SIMD)", speedup_seq });
    // std.debug.print("{s:<30}: {d:>12.2}x\n", .{ "Speedup (Optimized/SIMD)", speedup_optimized });

    // // 3. 검증 (두 결과가 수학적으로 같은지 확인)
    // std.debug.print("\n--- Validation (Tolerance: 1e-5) ---\n", .{});

    // // Base vs SIMD
    // var max_diff: f32 = 0;
    // for (result_base, 0..) |n, i| {
    //     const diff_re = @abs(n.re - result_simd[i].re);
    //     const diff_im = @abs(n.im - result_simd[i].im);
    //     if (diff_re > max_diff) max_diff = diff_re;
    //     if (diff_im > max_diff) max_diff = diff_im;
    // }
    // std.debug.print("{s:<30}: Diff={e:<10} | {s}\n", .{ "Validation (Base/SIMD)", max_diff, if (max_diff < 1e-5) "PASSED ✅" else "FAILED ❌" });

    // // Normal vs SIMD
    // max_diff = 0;
    // for (result_normal, 0..) |n, i| {
    //     const diff_re = @abs(n.re - result_simd[i].re);
    //     const diff_im = @abs(n.im - result_simd[i].im);
    //     if (diff_re > max_diff) max_diff = diff_re;
    //     if (diff_im > max_diff) max_diff = diff_im;
    // }
    // std.debug.print("{s:<30}: Diff={e:<10} | {s}\n", .{ "Validation (Normal/SIMD)", max_diff, if (max_diff < 1e-5) "PASSED ✅" else "FAILED ❌" });

    // // Sequential vs SIMD
    // max_diff = 0;
    // for (result_seq, 0..) |n, i| {
    //     const diff_re = @abs(n.re - result_simd[i].re);
    //     const diff_im = @abs(n.im - result_simd[i].im);
    //     if (diff_re > max_diff) max_diff = diff_re;
    //     if (diff_im > max_diff) max_diff = diff_im;
    // }
    // std.debug.print("{s:<30}: Diff={e:<10} | {s}\n", .{ "Validation (Seq/SIMD)", max_diff, if (max_diff < 1e-5) "PASSED ✅" else "FAILED ❌" });

    // // Optimized vs SIMD
    // max_diff = 0;
    // for (result_optimized, 0..) |n, i| {
    //     const diff_re = @abs(n.re - result_simd[i].re);
    //     const diff_im = @abs(n.im - result_simd[i].im);
    //     if (diff_re > max_diff) max_diff = diff_re;
    //     if (diff_im > max_diff) max_diff = diff_im;
    // }
    // std.debug.print("{s:<30}: Diff={e:<10} | {s}\n", .{ "Validation (Opt/SIMD)", max_diff, if (max_diff < 1e-5) "PASSED ✅" else "FAILED ❌" });

    // 1. 초기 데이터 준비 (len = 3)
    // var a = [_]Complex{
    //     .{ .re = 1.0, .im = 1.0 },
    //     .{ .re = 2.0, .im = 2.0 },
    //     .{ .re = 3.0, .im = 3.0 },
    // };
    // var b = [_]Complex{
    //     .{ .re = 1.0, .im = -1.0 },
    //     .{ .re = 1.0, .im = -1.0 },
    //     .{ .re = 1.0, .im = -1.0 },
    // };
    // var result = [_]Complex{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) };

    // // 2. 테스트 옵션 설정
    // const opt = Complex.convOptions{
    //     .mode = .mul,
    //     .a = .{ .conj = true, .scale = 2.0 }, // A를 켤레 취하고 2배 스케일링
    //     .acc_index = true, // 기존 result에 더함
    //     .acc_total = true, // 전체 합산 반환
    // };

    // // 3. 실행 시간 측정 시작
    // var start = try std.time.Timer.start();

    // // 4. 함수 실행
    // const total_sum = try Complex.generalComplexOpSIMD(&a, &b, &result, opt);
    // const end = start.read();

    // // 5. 결과 출력
    // std.debug.print("--- Test Results (len=3) ---\n", .{});
    // for (result, 0..) |res, i| {
    //     std.debug.print("Result[{d}]: {d:.2} + {d:.2}i\n", .{ i, res.re, res.im });
    // }
    // std.debug.print("Total Sum: {d:.2} + {d:.2}i\n", .{ total_sum.re, total_sum.im });
    // std.debug.print("Execution Time: {d} ns\n", .{end});

    // 덧셈 테스트
    // const opt_add = Complex.convOptions{
    //     .mode = .add,
    //     .a = .{ .conj = true, .scale = 2.0 },
    //     .b = .{ .conj = true, .scale = 0.5 },
    //     .out = .{ .conj = true, .scale = 1.0 },
    //     .acc_total = true,
    // };
    // const sum_add = try Complex.generalComplexOpSIMD(&a, &b, &result, opt_add);
    // std.debug.print("Add Sum: {d:.2} + {d:.2}i\n", .{ sum_add.re, sum_add.im });

    // // 뺄셈 테스트
    // result = [_]Complex{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) };
    // const opt_sub = Complex.convOptions{
    //     .mode = .sub,
    //     .a = .{ .conj = true, .scale = 2.0 },
    //     .b = .{ .conj = true, .scale = 0.5 },
    //     .out = .{ .conj = true, .scale = 1.0 },
    //     .acc_total = true,
    // };
    // const sum_sub = try Complex.generalComplexOpSIMD(&a, &b, &result, opt_sub);
    // std.debug.print("Sub Sum: {d:.2} + {d:.2}i\n", .{ sum_sub.re, sum_sub.im });

    // // 나눗셈 테스트
    // result = [_]Complex{ Complex.init(0, 0), Complex.init(0, 0), Complex.init(0, 0) };
    // const opt_div = Complex.convOptions{
    //     .mode = .div,
    //     .a = .{ .conj = true, .scale = 2.0 },
    //     .b = .{ .conj = true, .scale = 0.5 },
    //     .out = .{ .conj = true, .scale = 1.0 },
    //     .acc_total = true,
    // };
    // const sum_div = try Complex.generalComplexOpSIMD(&a, &b, &result, opt_div);
    // std.debug.print("Div Sum: {d:.2} + {d:.2}i\n", .{ sum_div.re, sum_div.im });

    // 1. 설정값 (여기서 시퀀스 길이를 조절하세요!)
    const seq_len: usize = 100000000;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // 2. 메모리 할당
    const a = try allocator.alloc(Complex, seq_len);
    const b = try allocator.alloc(Complex, seq_len);
    const res_simd = try allocator.alloc(Complex, seq_len);
    const res_scalar = try allocator.alloc(Complex, seq_len);
    defer allocator.free(a);
    defer allocator.free(b);
    defer allocator.free(res_simd);
    defer allocator.free(res_scalar);

    // 3. 랜덤 데이터 생성 (A, B)
    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed)); // OS에서 안전한 랜덤 시드를 가져옵니다.
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (0..seq_len) |i| {
        a[i] = .{ .re = rand.float(f32) * 10, .im = rand.float(f32) * 10 };
        b[i] = .{ .re = rand.float(f32) * 10, .im = rand.float(f32) * 10 };
        res_simd[i] = Complex.init(0, 0);
        res_scalar[i] = Complex.init(0, 0);
    }

    const opt = Complex.convOptions{
        .mode = .mul, // mul, add, sub, div 중 선택
        .a = .{ .conj = true, .scale = 1.5 },
        .b = .{ .conj = false, .scale = 0.8 },
        .out = .{ .conj = true, .scale = 2.0 },
        .acc_total = true,
    };

    std.debug.print("Benchmarking with sequence length: {d}\n\n", .{seq_len});

    // --- 일반 루프 측정 ---
    var timer = try std.time.Timer.start();
    const total_scalar = try Complex.generalComplexOp(a, b, res_scalar, opt);
    const time_scalar = timer.read();

    // // --- 캐시 플러싱 (중간에 끼워넣기) ---
    // {
    //     const dummy_size = 1024 * 1024 * 16; // 약 64MB
    //     const dummy = try allocator.alloc(f32, dummy_size);
    //     defer allocator.free(dummy);
    //     @memset(dummy, 1.0);

    //     var s: f32 = 0;
    //     for (dummy) |val| {
    //         s += val; // val을 사용해서 "unused variable" 에러 방지
    //     }
    //     std.mem.doNotOptimizeAway(s); // 컴파일러가 루프를 삭제하지 못하게 고정
    // }

    // --- SIMD 루프 측정 ---
    // var timer = try std.time.Timer.start();
    // const total_simd = try Complex.generalComplexOpSIMD(a, b, res_simd, opt);
    // const time_simd = timer.read();

    // 4. 결과 출력 및 오차 비교
    // const diff_re = @abs(total_simd.re - total_scalar.re);
    // const diff_im = @abs(total_simd.im - total_scalar.im);

    // std.debug.print("1. SIMD Results:\n", .{});
    // std.debug.print("   - Time: {d:>10} ns ({d:.2} ms)\n", .{ time_simd, @as(f64, @floatFromInt(time_simd)) / 1_000_000.0 });
    // std.debug.print("   - Sum : {d:.4} + {d:.4}i\n\n", .{ total_simd.re, total_simd.im });

    std.debug.print("2. Scalar Results:\n", .{});
    std.debug.print("   - Time: {d:>10} ns ({d:.2} ms)\n", .{ time_scalar, @as(f64, @floatFromInt(time_scalar)) / 1_000_000.0 });
    std.debug.print("   - Sum : {d:.4} + {d:.4}i\n\n", .{ total_scalar.re, total_scalar.im });

    // std.debug.print("3. Comparison:\n", .{});
    // std.debug.print("   - Speedup: {d:.2}x\n", .{@as(f64, @floatFromInt(time_scalar)) / @as(f64, @floatFromInt(time_simd))});
    // std.debug.print("   - Total Sum Error: Re({e}), Im({e})\n", .{ diff_re, diff_im });

    // const vectorSize: usize = std.simd.suggestVectorLength(f32) orelse 4;
    // std.debug.print("Vector size: {d}\n", .{vectorSize});
}
