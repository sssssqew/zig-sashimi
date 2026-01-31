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

fn prepareFFTInputs(buffer: []Complex, inputs: []const Complex) void {
    // 3. 데이터 복사 및 패딩
    @memcpy(buffer[0..inputs.len], inputs);
    @memset(buffer[inputs.len..], Complex.init(0, 0));
}
fn reverseBits(index: usize, n: usize) usize {
    var result: usize = 0;
    var tempIndex = index;
    var count = std.math.log2(n); // 몇 비트인지 계산 (8이면 3비트)

    while (count > 0) : (count -= 1) {
        result = (result << 1) | (tempIndex & 1);
        tempIndex >>= 1;
    }
    return result;
}
fn rearrange(data: []Complex) void {
    const n = data.len;
    for (0..n) |i| {
        const j = reverseBits(i, n);
        if (i < j) {
            const temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}
fn fft(data: []Complex) void {
    const n: usize = data.len;
    rearrange(data);
    var step: usize = 2;
    while (step <= n) : (step *= 2) {
        var start: usize = 0;
        while (start < n) : (start += step) {
            var k: usize = 0;
            const half = step / 2;
            while (k < half) : (k += 1) {
                const a = data[start + k];
                const b = data[start + k + half];
                const thetha = -2.0 * std.math.pi * @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(step));
                const w = Complex.init(std.math.cos(thetha), std.math.sin(thetha));
                const temp = w.mul(b);
                data[start + k] = a.add(temp);
                data[start + k + half] = a.sub(temp);
            }
        }
    }
}
fn ifft(data: []Complex) void {
    const n: usize = data.len;
    rearrange(data);
    var step: usize = 2;
    while (step <= n) : (step *= 2) {
        var start: usize = 0;
        while (start < n) : (start += step) {
            var k: usize = 0;
            const half = step / 2;
            while (k < half) : (k += 1) {
                const a = data[start + k];
                const b = data[start + k + half];
                const thetha = 2.0 * std.math.pi * @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(step));
                const w = Complex.init(std.math.cos(thetha), std.math.sin(thetha));
                const temp = w.mul(b);
                data[start + k] = a.add(temp);
                data[start + k + half] = a.sub(temp);
            }
        }
    }
    for (data, 0..) |_, i| {
        data[i].re /= @as(f32, @floatFromInt(n));
        data[i].im /= @as(f32, @floatFromInt(n));
    }
}

fn forwardConv(layer: *S4Layer, inputs: []const Complex) !void {
    @memset(layer.output_buffer, Complex.init(0.0, 0.0)); // Reset output for accumulation

    // 2. 커널(필터) 생성 및 패딩
    // layer.kernels는 이미 [N]Complex 형태여야 합니다.
    // (시퀀스 길이에 맞춰 커널을 미리 생성해두는 함수가 필요할 수 있습니다)
    for (layer.a_bars, 0..) |_, n| {
        prepareFFTInputs(layer.fft_result_buffer, layer.kernels[n]);

        fft(layer.fft_result_buffer);
        Complex.mulSIMD(layer.fft_input_buffer, layer.fft_result_buffer, layer.fft_result_buffer);
        ifft(layer.fft_result_buffer); // layer.fft_result_buffer: 특정 채널의 출력신호
        Complex.addSIMD(layer.fft_result_buffer[0..inputs.len], layer.output_buffer, layer.output_buffer);
    }
}
pub fn predict(layer: *S4Layer, inputs: []const Complex) !void {
    // 1. 입력 패딩
    prepareFFTInputs(layer.fft_input_buffer, inputs);
    // 입력 신호를 주파수 영역으로 변환
    fft(layer.fft_input_buffer);

    // 1. FFT 기반 (CNN 모드) 결과 얻기
    try forwardConv(layer, inputs);
}

/// 1. 메모리 효율을 위해 끊어서 학습
pub fn trainTruncatedBPTT(layer: *S4Layer, inputs: []const Complex, targets: []const Complex, config: TrainConfig) !void {
    // layer.a_bars, layer.b_bars 등을 업데이트하는 방식으로 구현
    const n_channels = layer.a_bars.len;
    const inv_len = 1.0 / @as(f32, @floatFromInt(config.window_size));

    // 1. 루프 시작 전 (또는 함수 초반) 메모리 할당
    const total_grad_a = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_b = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_c = try layer.allocator.alloc(Complex, n_channels);
    const prevStates = try layer.allocator.alloc(Complex, n_channels);
    const params = struct { da_bar_da: Complex, db_bar_db: Complex, db_bar_da: Complex };
    var paramsOfChannels = try layer.allocator.alloc(params, n_channels);
    defer {
        layer.allocator.free(total_grad_a);
        layer.allocator.free(total_grad_b);
        layer.allocator.free(total_grad_c);
        layer.allocator.free(prevStates);
        layer.allocator.free(paramsOfChannels);
    }

    // Training Loop: Optimizing via Backpropagation Through Time (BPTT)
    for (0..config.epochs) |epoch| {
        @memset(layer.states, Complex.init(0, 0));
        var start_idx: usize = 0;

        while (start_idx < inputs.len) : (start_idx += config.window_size) {
            var total_loss: f32 = 0;
            // 윈도우 시작할 때마다 0으로 초기화
            @memset(total_grad_a, Complex.init(0, 0));
            @memset(total_grad_b, Complex.init(0, 0));
            @memset(total_grad_c, Complex.init(0, 0));

            const end_idx = @min(start_idx + config.window_size, inputs.len);

            for (0..n_channels) |n| {
                const common_denom = Complex.init(2.0, 0).sub(layer.a_continuous[n].scale(layer.dt));
                paramsOfChannels[n] = .{ .da_bar_da = try Complex.init(4.0 * layer.dt, 0).div(common_denom.square()), .db_bar_db = try Complex.init(2.0 * layer.dt, 0).div(common_denom), .db_bar_da = try Complex.init(2.0 * layer.dt * layer.dt, 0).mul(layer.b_continuous[n]).div(common_denom.square()) };
            }

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
                    const grad_a_bar = err.mul(layer.c_coeffs[n].conj()).mul(prevStates[n].conj());
                    const grad_b_bar = err.mul(layer.c_coeffs[n].conj()).mul(u.conj());

                    total_grad_a[n] = total_grad_a[n].add(grad_a_bar.mul(paramsOfChannels[n].da_bar_da).add(grad_b_bar.mul(paramsOfChannels[n].db_bar_da)));
                    total_grad_b[n] = total_grad_b[n].add(grad_b_bar.mul(paramsOfChannels[n].db_bar_db));
                    total_grad_c[n] = total_grad_c[n].add(err.mul(layer.states[n].conj()));
                }
            }

            // Global Parameter Update once per epoch
            for (0..n_channels) |n| {
                // Applying Gradient Descent
                layer.a_continuous[n] = layer.a_continuous[n].sub(total_grad_a[n].scale(config.lr_a * inv_len));
                layer.b_continuous[n] = layer.b_continuous[n].sub(total_grad_b[n].scale(config.lr_b * inv_len));
                layer.c_coeffs[n] = layer.c_coeffs[n].sub(total_grad_c[n].scale(config.lr_c * inv_len));

                // Spectral Radius Constraint: Enforce system stability (A(re) < 0)
                if (layer.a_continuous[n].re > -1e-4) layer.a_continuous[n].re = -1e-4;
                if (epoch % 50 == 0) {
                    std.debug.print("Epoch {d} Channel {d}: Loss = {d:.6}, A = {d:.3} + {d:.3}i B = {d:.3} + {d:.3}i C = {d:.3} + {d:.3}i\n", .{ epoch, n, total_loss, layer.a_bars[n].re, layer.a_bars[n].im, layer.b_bars[n].re, layer.b_bars[n].im, layer.c_coeffs[n].re, layer.c_coeffs[n].im });
                }
            }
            // 커널 업데이트: 학습된 a, b, c를 다시 컨볼루션 커널로 변환 (추론을 위해)
            try layer.updateDiscretizedParams();
        }
    }
}

// 2. 정석적인 전체 시퀀스 학습 (정답지)
pub fn trainFullBPTT(layer: *S4Layer, inputs: []const Complex, targets: []const Complex, config: TrainConfig) !void {
    const n_channels = layer.a_bars.len;
    const seq_len = inputs.len;
    const inv_len = 1.0 / @as(f32, @floatFromInt(seq_len));

    // 1. [시퀀스 길이 + 1]만큼의 상태를 저장할 거대한 단일 버퍼 할당
    // (t=0일 때의 초기 상태가 필요하므로 seq_len + 1입니다)
    const history_size = (seq_len + 1) * n_channels;
    const history = try layer.allocator.alloc(Complex, history_size);
    defer layer.allocator.free(history);

    // 2. 인덱싱 헬퍼 함수 (특정 시점 t의 상태 슬라이스를 반환)
    // 이 함수를 정의해두면 history[t] 처럼 접근하기 편합니다.
    const getState = struct {
        fn get(buf: []Complex, t: usize, n: usize, channels: usize) *Complex {
            const index = (t * channels) + n;
            return &buf[index];
        }
    }.get;

    // 1. 루프 시작 전 (또는 함수 초반) 메모리 할당
    const total_grad_a = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_b = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_c = try layer.allocator.alloc(Complex, n_channels);

    const params = struct { da_bar_da: Complex, db_bar_db: Complex, db_bar_da: Complex };
    var paramsOfChannels = try layer.allocator.alloc(params, n_channels);
    var next_state_grad = try layer.allocator.alloc(Complex, n_channels);
    defer {
        layer.allocator.free(total_grad_a);
        layer.allocator.free(total_grad_b);
        layer.allocator.free(total_grad_c);
        layer.allocator.free(paramsOfChannels);
        layer.allocator.free(next_state_grad);
    }

    for (0..config.epochs) |epoch| {
        // 1. 매 에포크 시작 시 그라디언트 및 히스토리 초기화
        @memset(history, Complex.init(0, 0));
        @memset(total_grad_a, Complex.init(0, 0));
        @memset(total_grad_b, Complex.init(0, 0));
        @memset(total_grad_c, Complex.init(0, 0));
        @memset(next_state_grad, Complex.init(0, 0));

        // 2. 현재 파라미터(A, B)에 기반한 이산화 미분 계수 계산
        for (0..n_channels) |n| {
            const common_denom = Complex.init(2.0, 0).sub(layer.a_continuous[n].scale(layer.dt));
            paramsOfChannels[n] = .{ .da_bar_da = try Complex.init(4.0 * layer.dt, 0).div(common_denom.square()), .db_bar_db = try Complex.init(2.0 * layer.dt, 0).div(common_denom), .db_bar_da = try Complex.init(2.0 * layer.dt * layer.dt, 0).mul(layer.b_continuous[n]).div(common_denom.square()) };
        }
        // 3. Forward Pass (상태 기록)
        for (inputs, 0..) |u, t| {
            for (0..n_channels) |n| {
                // 1. 직전 상태 가져오기 (t 시점)
                const prevX = getState(history, t, n, n_channels).*;
                // 2. 이산화된 수식 적용: x_{t+1} = A_bar * x_t + B_bar * u_t
                const ax = prevX.mul(layer.a_bars[n]);
                const bu = u.mul(layer.b_bars[n]);
                const nextX = ax.add(bu);

                // 3. 다음 상태 저장하기 (t + 1 시점)
                getState(history, t + 1, n, n_channels).* = nextX;
            }
        }
        // 4. Backward Pass 시작 (역방향 루프)

        // 다음 시점에서 넘어오는 상태 그라디언트를 보관할 버퍼 (초기값 0)
        // dL/dx_{t+1} 값을 들고 다음 루프로 넘겨주는 역할을 합니다.

        var t: usize = seq_len;
        var total_loss: f32 = 0;
        while (t > 0) : (t -= 1) {
            const idx = t - 1; // 입력/타겟 인덱스 (0 ~ seq_len-1)
            const target = targets[idx];

            // 현재 시점(t)의 상태를 history에서 가져옵니다.
            // 주의: t=0은 초기상태이므로, t=1부터 실제 출력이 나옵니다.

            var y = Complex.init(0, 0); // 최종 출력계산
            for (0..n_channels) |n| {
                const xt = getState(history, t, n, n_channels).*;
                y = y.add(xt.mul(layer.c_coeffs[n]));
            }
            // 오차 계산 (Loss Gradient: dL/dy)
            const err = y.sub(target);
            const loss = err.re * err.re + err.im * err.im;
            total_loss += loss;

            for (0..n_channels) |n| {
                const xt = getState(history, t, n, n_channels).*;
                const prevX = getState(history, t - 1, n, n_channels).*;
                const u = inputs[idx];

                // 1. Output Gradient (dL/dC)
                const grad_c = err.mul(xt.conj());
                total_grad_c[n] = total_grad_c[n].add(grad_c);

                // 2. State Gradient (dL/dx_t)
                // 현재 오차로부터 오는 것 + 다음 시점(t+1)에서 역전파되어 온 것
                const current_x_grad = err.mul(layer.c_coeffs[n].conj()).add(next_state_grad[n]);

                // 3. Parameter Gradients (A_bar, B_bar)
                const grad_a_bar = current_x_grad.mul(prevX.conj());
                const grad_b_bar = current_x_grad.mul(u.conj());

                // 4. 이산화 수식(Bilinear)을 고려한 최종 파라미터(A, B) 그라디언트 누적
                total_grad_a[n] = total_grad_a[n].add(grad_a_bar.mul(paramsOfChannels[n].da_bar_da).add(grad_b_bar.mul(paramsOfChannels[n].db_bar_da)));
                total_grad_b[n] = total_grad_b[n].add(grad_b_bar.mul(paramsOfChannels[n].db_bar_db));

                // 5. Next Step을 위한 State Gradient 전파 (dL/dx_{t-1})
                // dx_t/dx_{t-1} = A_bar 이므로
                next_state_grad[n] = current_x_grad.mul(layer.a_bars[n].conj());
            }
        }
        // 5. 가중치 업데이트 (GD) 및 시스템 안정화
        for (0..n_channels) |n| {
            layer.a_continuous[n] = layer.a_continuous[n].sub(total_grad_a[n].scale(config.lr_a * inv_len));
            layer.b_continuous[n] = layer.b_continuous[n].sub(total_grad_b[n].scale(config.lr_b * inv_len));
            layer.c_coeffs[n] = layer.c_coeffs[n].sub(total_grad_c[n].scale(config.lr_c * inv_len));

            // 안정성 제약 (Spectral Radius Constraint)
            // S4의 핵심: 시스템이 발산하지 않도록 Re(A) < 0을 강제합니다.
            if (layer.a_continuous[n].re > -1e-4) layer.a_continuous[n].re = -1e-4;
            if (epoch % 100 == 0) {
                std.debug.print("Epoch {d} Channel {d}: Loss = {d:.6}, A = {d:.3} + {d:.3}i B = {d:.3} + {d:.3}i C = {d:.3} + {d:.3}i\n", .{ epoch, n, total_loss, layer.a_bars[n].re, layer.a_bars[n].im, layer.b_bars[n].re, layer.b_bars[n].im, layer.c_coeffs[n].re, layer.c_coeffs[n].im });
            }
        }
        // 6. 커널 업데이트: 학습된 a, b, c를 바탕으로 a_bar, b_bar를 다시 계산하고 컨볼루션 커널로 변환 (추론을 위해)
        try layer.updateDiscretizedParams();
    }
    // 학습 종료 후 최종 커널 한 번만 생성
    try layer.setupKernels();
}

// 3. 대규모 가속 학습
pub fn trainConv(layer: *S4Layer, inputs: []const Complex, targets: []const Complex, config: TrainConfig) !void {
    const n_channels = layer.a_bars.len;
    const seq_len = inputs.len;
    const inv_len = 1.0 / @as(f32, @floatFromInt(seq_len));

    // 1. 루프 시작 전 (또는 함수 초반) 메모리 할당
    const total_grad_a = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_b = try layer.allocator.alloc(Complex, n_channels);
    const total_grad_c = try layer.allocator.alloc(Complex, n_channels);

    const params = struct { da_bar_da: Complex, db_bar_db: Complex, db_bar_da: Complex };
    var paramsOfChannels = try layer.allocator.alloc(params, n_channels);

    defer {
        layer.allocator.free(total_grad_a);
        layer.allocator.free(total_grad_b);
        layer.allocator.free(total_grad_c);
        layer.allocator.free(paramsOfChannels);
    }

    std.debug.assert(seq_len <= layer.output_buffer.len);
    if (seq_len > layer.output_buffer.len) return error.InputLengthMisMatch;

    // 1. 입력 패딩 및 입력 신호를 주파수 영역으로 변환
    prepareFFTInputs(layer.fft_input_buffer, inputs);
    fft(layer.fft_input_buffer);

    for (0..config.epochs) |epoch| {
        @memset(total_grad_a, Complex.init(0, 0));
        @memset(total_grad_b, Complex.init(0, 0));
        @memset(total_grad_c, Complex.init(0, 0));

        // 2. 현재 파라미터(A, B)에 기반한 이산화 미분 계수 계산
        for (0..n_channels) |n| {
            const common_denom = Complex.init(2.0, 0).sub(layer.a_continuous[n].scale(layer.dt));
            paramsOfChannels[n] = .{ .da_bar_da = try Complex.init(4.0 * layer.dt, 0).div(common_denom.square()), .db_bar_db = try Complex.init(2.0 * layer.dt, 0).div(common_denom), .db_bar_da = try Complex.init(2.0 * layer.dt * layer.dt, 0).mul(layer.b_continuous[n]).div(common_denom.square()) };
        }

        // 1. FFT 기반 (CNN 모드) 결과 얻기
        try forwardConv(layer, inputs); // layer.output_buffer : 최종출력
        // 2. 전체오차 계산
        const total_loss = Complex.totalLossSIMD(layer.output_buffer, targets);
        Complex.subSIMD(layer.output_buffer, targets, layer.output_buffer); // layer.output_buffer : 오차신호

        // 여기서 정규화 추가 (SIMD를 쓰지 않아도 시퀀스 1000개 정도는 이 루프가 매우 빠릅니다)
        for (layer.output_buffer) |*err| {
            err.* = err.scale(inv_len);
        }
        // 3. 역전파

        // 커널 기울기 구하기

        // // fft 결과를 켤레 복소수로 변환
        // Complex.conjSIMD(paddedInputs);

        prepareFFTInputs(layer.fft_result_buffer, layer.output_buffer); // 커널 기울기

        // 오차 신호를 주파수 영역으로 변환
        fft(layer.fft_result_buffer);

        Complex.mulSIMDWithConj(layer.fft_result_buffer, layer.fft_input_buffer, layer.fft_result_buffer);

        // 시간 영역으로 복귀
        ifft(layer.fft_result_buffer); // layer.fft_result_buffer : 커널 기울기

        // 2. 채널별 파라미터 업데이트 루프
        for (0..layer.a_bars.len) |n| {
            // var grad_a = Complex.init(0, 0);
            // var grad_b = Complex.init(0, 0);
            // var grad_c = Complex.init(0, 0);

            var a_pow = Complex.init(1, 0);
            var a_pow_prev = Complex.init(1, 0);

            for (0..seq_len) |i| {
                const dL_dK = layer.fft_result_buffer[i]; // 공통의 커널 기울기

                // 각 채널 n의 현재 파라미터 상태를 이용해 개별 기울기 추출
                // C_n 미분
                const grad_c = dL_dK.mul(a_pow.mul(layer.b_bars[n]));
                // B_n 미분
                const grad_b = dL_dK.mul(layer.c_coeffs[n].mul(a_pow));
                // A_n 미분
                var grad_a = Complex.init(0, 0);
                if (i > 0) {
                    const i_f = @as(f32, @floatFromInt(i));
                    const term_a = Complex.init(i_f, 0).mul(layer.c_coeffs[n]).mul(layer.b_bars[n]).mul(a_pow_prev);
                    grad_a = dL_dK.mul(term_a);
                }

                a_pow_prev = a_pow;
                a_pow = a_pow.mul(layer.a_bars[n]);

                // 4. 이산화 수식(Bilinear)을 고려한 최종 파라미터(A, B) 그라디언트 누적
                total_grad_a[n] = total_grad_a[n].add(grad_a.mul(paramsOfChannels[n].da_bar_da).add(grad_b.mul(paramsOfChannels[n].db_bar_da)));
                total_grad_b[n] = total_grad_b[n].add(grad_b.mul(paramsOfChannels[n].db_bar_db));
                total_grad_c[n] = total_grad_c[n].add(grad_c);
            }

            // 3. 파라미터 업데이트 (SGD)
            // layer.c_coeffs[n] = layer.c_coeffs[n].sub(grad_c.mul(config.lr_c));
            // layer.b_bars[n] = layer.b_bars[n].sub(grad_b.mul(config.lr_b));
            // layer.a_bars[n] = layer.a_bars[n].sub(grad_a.mul(config.lr_a));

            layer.a_continuous[n] = layer.a_continuous[n].sub(total_grad_a[n].scale(config.lr_a));
            layer.b_continuous[n] = layer.b_continuous[n].sub(total_grad_b[n].scale(config.lr_b));
            layer.c_coeffs[n] = layer.c_coeffs[n].sub(total_grad_c[n].scale(config.lr_c));

            // 안정성 제약 (Spectral Radius Constraint)
            // S4의 핵심: 시스템이 발산하지 않도록 Re(A) < 0을 강제합니다.
            if (layer.a_continuous[n].re > -0.1) layer.a_continuous[n].re = -0.1;
            if (epoch % 100 == 0) {
                std.debug.print("Epoch {d} Channel {d}: Loss = {d:.6}, A = {d:.3} + {d:.3}i B = {d:.3} + {d:.3}i C = {d:.3} + {d:.3}i\n", .{ epoch, n, total_loss, layer.a_bars[n].re, layer.a_bars[n].im, layer.b_bars[n].re, layer.b_bars[n].im, layer.c_coeffs[n].re, layer.c_coeffs[n].im });
            }
        }
        // 커널 업데이트: 학습된 a, b, c를 다시 컨볼루션 커널로 변환 (추론을 위해)
        try layer.updateDiscretizedParams();
        try layer.setupKernels();
    }
}
