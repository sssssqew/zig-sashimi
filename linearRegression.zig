const std = @import("std");
const persistence = @import("persistence.zig");

/// Hyperparameters and control flow options for the training process.
pub const TrainOptions = struct {
    epochs: usize = 1000,
    learningRate: f32 = 0.01,
    silent: bool = false,
    logInterval: usize = 1000,
    targetLoss: ?f32 = null,
};

/// A snapshot of the model state for serialization and recovery.
pub const Checkpoint = struct {
    weights: []f32,
    bias: f32,
    epochs: usize,
    lr: f32,
};

/// Simple Regression Head with SIMD-accelerated inference and Gradient Descent.
pub const RegressionHead = struct {
    weights: []f32,
    bias: f32 = 0.0,
    /// Tracking flag for memory-mapped storage to ensure safe resource deallocation.
    isMmaped: bool = false,
    mmapRaw: []u8 = &.{}, // Stores the original byte slice returned by mmap to ensure correct deallocation.

    /// Initializes weights with a uniform random distribution.
    pub fn init(allocator: std.mem.Allocator, inputLen: usize) !RegressionHead {
        const weights = try allocator.alloc(f32, inputLen);

        // Deterministic seeding for reproducible weight initialization.
        var prng = std.Random.DefaultPrng.init(12345);
        const random = prng.random();

        for (weights) |*w| {
            w.* = random.float(f32) * 0.1;
        }
        return .{ .weights = weights };
    }

    pub fn deinit(self: *RegressionHead, allocator: std.mem.Allocator) void {
        self.freeWeights(allocator);
    }

    /// Performs forward pass using SIMD (Single Instruction, Multiple Data) vectorization.
    /// Exploits hardware-level parallelism for dot product calculation.
    pub fn predict(self: *const RegressionHead, input: []const f32) f32 {
        std.debug.assert(input.len == self.weights.len);

        const len = input.len;
        // Determine optimal vector width for the target architecture (SSE/AVX/NEON).
        const vectorSize = std.simd.suggestVectorLength(f32) orelse 4;
        var sumVector: @Vector(vectorSize, f32) = @splat(0.0);

        var i: usize = 0;
        // Primary SIMD processing loop.
        while (i + vectorSize <= len) : (i += vectorSize) {
            const vInput: @Vector(vectorSize, f32) = input[i..][0..vectorSize].*;
            const vWeight: @Vector(vectorSize, f32) = self.weights[i..][0..vectorSize].*;
            sumVector += vInput * vWeight;
        }

        // Horizontal addition of vector lanes.
        var totalSum: f32 = @reduce(.Add, sumVector);

        // Scalar fallback for remaining elements (tail handling).
        while (i < len) : (i += 1) {
            totalSum += input[i] * self.weights[i];
        }

        return totalSum + self.bias;
    }

    /// Computes the Squared Error loss for the given prediction.
    fn calcLoss(self: *const RegressionHead, prediction: f32, target: f32) f32 {
        _ = self;
        const diff = prediction - target;
        return diff * diff;
    }

    /// Updates model parameters using Stochastic Gradient Descent (SGD).
    fn update(self: *RegressionHead, input: []const f32, prediction: f32, target: f32, lr: f32) void {
        const err = prediction - target;
        for (self.weights, input) |*w, x| {
            w.* -= lr * err * x;
        }
        self.bias -= lr * err;
    }

    /// Primary training loop with periodic logging and early stopping support.
    pub fn train(self: *RegressionHead, inputs: []const []const f32, targets: []const f32, options: TrainOptions) !void {
        var e: usize = 0;
        while (e < options.epochs) : (e += 1) {
            var totalLoss: f32 = 0;
            for (inputs, targets) |input, target| {
                const p = self.predict(input);
                totalLoss += self.calcLoss(p, target);
                self.update(input, p, target, options.learningRate);
            }
            const avgLoss = totalLoss / @as(f32, @floatFromInt(targets.len));

            if (!options.silent and e % options.logInterval == 0) {
                std.debug.print("Epoch {d}: Avg Loss = {d:.6}\n", .{ e, avgLoss });
            }

            if (options.targetLoss) |tl| {
                if (avgLoss < tl) {
                    std.debug.print("Training converged. Target loss reached.\n", .{});
                    break;
                }
            }

            // Auto-checkpointing for long-running training sessions.
            if (e > 0 and e % options.logInterval == 0) {
                try self.saveModel("Train_temp.bin", e, options.learningRate);
            }
        }
    }

    /// Serializes the current model state to the filesystem.
    pub fn saveModel(self: *const RegressionHead, filename: []const u8, epochs: usize, lr: f32) !void {
        const cp = Checkpoint{
            .weights = self.weights,
            .bias = self.bias,
            .epochs = epochs,
            .lr = lr,
        };
        try persistence.Storage.save(filename, cp);
        std.debug.print("Model persisted: {s} ({d} weights)\n", .{ filename, self.weights.len });
    }

    /// Loads model state using standard heap allocation.
    pub fn loadModel(self: *RegressionHead, filename: []const u8, allocator: std.mem.Allocator) !Checkpoint {
        const cp = try persistence.Storage.load(allocator, filename, Checkpoint);
        self.updateFromCheckpoint(allocator, cp, false, &.{});
        std.debug.print("Model loaded from disk: {s}\n", .{filename});
        return cp;
    }

    /// High-performance model loading using zero-copy Memory Mapping (mmap).
    /// Provides near-instant access to large weight buffers.
    pub fn loadModelMmap(self: *RegressionHead, filename: []const u8, allocator: std.mem.Allocator) !Checkpoint {
        const mmapResult = try persistence.Storage.loadMmap(filename, Checkpoint);
        const cp = mmapResult.data;

        self.updateFromCheckpoint(allocator, cp, true, mmapResult.raw);
        std.debug.print("Model mmapped for high-speed access: {s}\n", .{filename});
        return cp;
    }

    /// Synchronizes the model state with a loaded checkpoint, handling memory transitions.
    fn updateFromCheckpoint(self: *RegressionHead, allocator: std.mem.Allocator, cp: Checkpoint, isMmap: bool, raw: []u8) void {
        self.freeWeights(allocator);
        std.debug.assert(self.weights.len == 0);

        self.weights = cp.weights;
        self.bias = cp.bias;
        self.isMmaped = isMmap;
        self.mmapRaw = raw;
    }

    /// Disposes of weight memory depending on the allocation strategy (Heap vs. Mmap).
    fn freeWeights(self: *RegressionHead, allocator: std.mem.Allocator) void {
        // 1. Prevent double-free by checking if weights are already deallocated.
        if (self.weights.len == 0) return;

        if (self.isMmaped) {
            // 2. Memory-mapped deallocation logic.
            if (self.mmapRaw.len > 0) {
                // Cast to a page-aligned slice as required by POSIX munmap.
                const alignedBytes: []align(std.heap.page_size_min) const u8 = @alignCast(self.mmapRaw);
                std.posix.munmap(alignedBytes);
            }
        } else {
            // 3. Standard heap deallocation logic.
            // If an error occurs here, it indicates an 'isMmaped' flag mismatch.
            allocator.free(self.weights);
        }

        // 4. Reset state to ensure the struct is in a clean, predictable state.
        // This prevents accidental reuse of deallocated resources.
        self.weights = &.{};
        self.mmapRaw = &.{};
        self.isMmaped = false;
        self.bias = 0.0;
    }
};
