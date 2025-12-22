const std = @import("std");
const LinearRegression = @import("linear_regression.zig");

// Define a data structure for a Fish
const Fish = struct {
    length: f32,
    width: f32,
};

// Simple Linear Regression logic: Predict weight based on length and width
fn predictWeight(fish: Fish) f32 {
    return (fish.length * 0.5) + (fish.width * 0.2);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // const input = try allocator.alloc(f32, 128);
    // defer allocator.free(input);
    // @memset(input, 0.5);

    // const target = 10.0;
    // const learningRate = 0.001;

    // std.debug.print("created weights: {any}\n", .{head.weights});

    // Mock data: A list of fish features
    const fishes = [_]Fish{
        .{ .length = 10.5, .width = 3.2 },
        .{ .length = 20.0, .width = 5.5 },
        .{ .length = 15.2, .width = 4.0 },
    };

    var inputs = try allocator.alloc([]f32, fishes.len);
    var targets = try allocator.alloc(f32, fishes.len);

    defer {
        for (inputs) |row| {
            allocator.free(row);
        }
        allocator.free(inputs);
        allocator.free(targets);
    }

    for (fishes, 0..) |f, i| {
        const row = try allocator.alloc(f32, 2);
        row[0] = f.length;
        row[1] = f.width;
        inputs[i] = row;
        targets[i] = predictWeight(f);
    }
    var head = try LinearRegression.RegressionHead.init(allocator, 2);
    // defer head.deinit(allocator);
    try head.train(inputs, targets, .{ .epochs = 30000, .learningRate = 0.001, .logInterval = 1000, .targetLoss = 0.00001, .silent = true });
    std.debug.print("length weight: {d:.5}, width weight: {d:.5}\n", .{ head.weights[0], head.weights[1] });
    try head.saveModel("Train.bin", 30000, 0.001);

    const loadedModelMmap = try head.loadModelMmap("Train.bin", allocator);

    // 주의: mmap은 allocator.free를 사용하지 않습니다.
    // (실제로는 std.posix.munmap을 써야 하지만, 간단한 테스트 시에는
    //  프로그램 종료 시 OS가 회수하도록 두어도 무방합니다.)

    std.debug.print("loaded Model (Mmap): {any}\n", .{loadedModelMmap});

    // const loadedModel = try head.loadModel("Train.bin", allocator);
    // defer {
    //     allocator.free(loadedModel.weights);
    // }
    // std.debug.print("loaded Model: {any}\n", .{loadedModel});

    const newFish: Fish = .{ .length = 25.0, .width = 7.0 };
    const inp = [_]f32{ newFish.length, newFish.width };
    const prediction = head.predict(&inp);
    const answer = predictWeight(newFish);
    std.debug.print("prediction: {d:.5}, answer = {d:.5}\n", .{ prediction, answer });
    const accuracy = (1.0 - @abs(prediction - answer) / answer) * 100.0;
    std.debug.print("Accuracy: {d:.2}%\n", .{accuracy});

    // 2. mmap 로드 테스트
    // allocator 인자가 없는 loadModelMmap을 호출합니다.
}
