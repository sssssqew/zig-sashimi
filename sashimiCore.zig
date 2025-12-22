const std = @import("std");
const LinearRegression = @import("linearRegression.zig");

// Define a data structure for a Fish
/// Representation of biological metrics for the synthetic dataset.
const Fish = struct {
    length: f32,
    width: f32,
};

// Simple Linear Regression logic: Predict weight based on length and width
/// Ground truth function used to generate target labels for supervised learning.
/// This serves as the reference function for reverse-engineering parameters.
fn predictWeight(fish: Fish) f32 {
    return (fish.length * 0.5) + (fish.width * 0.2);
}

pub fn main() !void {
    // Initialize General Purpose Allocator (GPA) to monitor memory safety and leaks.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // Ensure all allocated resources are released upon process termination.
    defer _ = gpa.deinit();

    // Mock data: A list of fish features
    const fishes = [_]Fish{
        .{ .length = 10.5, .width = 3.2 },
        .{ .length = 20.0, .width = 5.5 },
        .{ .length = 15.2, .width = 4.0 },
    };

    // Allocate buffers for feature vectors (inputs) and ground truth labels (targets).
    var inputs = try allocator.alloc([]f32, fishes.len);
    var targets = try allocator.alloc(f32, fishes.len);

    // RAII-style cleanup for heap-allocated slices and nested feature rows.
    defer {
        for (inputs) |row| {
            allocator.free(row);
        }
        allocator.free(inputs);
        allocator.free(targets);
    }

    // Populate training data by mapping structured metrics to vectorized representations.
    for (fishes, 0..) |f, i| {
        const row = try allocator.alloc(f32, 2);
        row[0] = f.length;
        row[1] = f.width;
        inputs[i] = row;
        targets[i] = predictWeight(f);
    }

    // Instantiate the RegressionHead engine with an input dimensionality of 2.
    var head = try LinearRegression.RegressionHead.init(allocator, 2);
    // Register mandatory deinitialization for internal weight buffers.
    defer head.deinit(allocator);

    // Execute Stochastic Gradient Descent (SGD) to optimize parameters towards target loss.
    try head.train(inputs, targets, .{ .epochs = 30000, .learningRate = 0.001, .logInterval = 1000, .targetLoss = 0.00001, .silent = true });

    // Log learned coefficients for architectural verification.
    std.debug.print("length weight: {d:.5}, width weight: {d:.5}\n", .{ head.weights[0], head.weights[1] });

    // Serialize the converged model state to a binary file for persistence.
    try head.saveModel("Train.bin", 30000, 0.001);

    // Performance testing of the POSIX mmap-based zero-copy loading mechanism.
    const loadedModelMmap = try head.loadModelMmap("Train.bin", allocator);
    std.debug.print("loaded Model (Mmap): {any}\n", .{loadedModelMmap});

    // NOTE: Memory-mapped regions (mmap) do not use allocator.free.
    // (While std.posix.munmap should be called for strict cleanup,
    // letting the OS reclaim it upon process termination is acceptable for simple tests.)

    // const loadedModel = try head.loadModel("Train.bin", allocator);
    // std.debug.print("loaded Model: {any}\n", .{loadedModel});

    // Validation Phase: Forward pass inference on unseen data to assess predictive accuracy.
    const newFish: Fish = .{ .length = 25.0, .width = 7.0 };
    const inp = [_]f32{ newFish.length, newFish.width };
    const prediction = head.predict(&inp);
    const answer = predictWeight(newFish);
    std.debug.print("prediction: {d:.5}, answer = {d:.5}\n", .{ prediction, answer });

    // Calculate error metrics to determine the model's convergence quality.
    const accuracy = (1.0 - @abs(prediction - answer) / answer) * 100.0;
    std.debug.print("Accuracy: {d:.2}%\n", .{accuracy});

    // 2. mmap load test
    // Invoking loadModelMmap without an explicit allocator argument for weight mapping.
}
