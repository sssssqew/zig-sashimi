const std = @import("std");

/// A high-performance binary serialization engine for persisting and recovering Zig data structures.
/// Implements a custom binary protocol with type-safety checks and alignment padding.
pub const Storage = struct {
    /// Global Magic Number for file format identification (Zig Data).
    const GLOBAL_MAGIC = "ZDAT";
    /// Internal protocol version to ensure backward and forward compatibility.
    const ENGINE_VERSION: u32 = 2;

    /// Serializes a data structure to a binary file.
    /// Includes metadata such as magic number, versioning, type size, and field counts for validation.
    pub fn save(filename: []const u8, data: anytype) !void {
        var file = try std.fs.cwd().createFile(filename, .{});
        defer file.close();

        const T = @TypeOf(data);

        // Header Section: Metadata for integrity verification
        try file.writeAll(GLOBAL_MAGIC);
        try writeAnyInt(file, u32, ENGINE_VERSION);
        try writeAnyInt(file, u32, @as(u32, @intCast(@sizeOf(T))));

        // Structural metadata for runtime validation
        const field_count: u32 = if (@typeInfo(T) == .@"struct") @intCast(@typeInfo(T).@"struct".fields.len) else 0;
        try writeAnyInt(file, u32, field_count);

        // Type reflection: Store type name for debugging and safety
        const type_name = @typeName(T);
        try writeAnyInt(file, u32, @intCast(type_name.len));
        try file.writeAll(type_name);

        // Payload Section: Recursive data serialization
        try writeValue(file, data);
    }

    /// Deserializes binary data from a file into a Zig data structure.
    /// Performs rigorous validation of metadata to prevent memory corruption or type mismatch.
    pub fn load(allocator: std.mem.Allocator, filename: []const u8, comptime T: type) !T {
        var file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        // Integrity Check: Verify File Signature
        var magic_buf: [4]u8 = undefined;
        try readFull(file, &magic_buf);
        if (!std.mem.eql(u8, &magic_buf, GLOBAL_MAGIC)) return error.InvalidMagicNumber;

        // Versioning Check: Prevent loading of future protocol versions
        const ver = try readAnyInt(file, u32);
        if (ver > ENGINE_VERSION) return error.UnsupportedVersion;

        // Type Consistency Check: Verify struct size and field parity
        const saved_size = try readAnyInt(file, u32);
        if (saved_size != @sizeOf(T)) return error.TypeSizeMismatch;

        const saved_fields = try readAnyInt(file, u32);
        if (@typeInfo(T) == .@"struct" and saved_fields != @typeInfo(T).@"struct".fields.len) {
            return error.StructFieldCountMismatch;
        }

        // Skip type name metadata segment
        const name_len = try readAnyInt(file, u32);
        try file.seekBy(@intCast(name_len));

        // Recursive deserialization into the target type
        return try readValue(allocator, file, T);
    }

    /// Helper for writing primitives with Little-Endian byte ordering.
    fn writeAnyInt(file: std.fs.File, comptime AnyT: type, value: AnyT) !void {
        var buf: [@sizeOf(AnyT)]u8 = undefined;
        switch (@typeInfo(AnyT)) {
            .int => std.mem.writeInt(AnyT, &buf, value, .little),
            .float => {
                const IntT = std.meta.Int(.unsigned, @sizeOf(AnyT) * 8);
                std.mem.writeInt(IntT, &buf, @bitCast(value), .little);
            },
            else => @compileError("Unsupported numeric type: " ++ @typeName(AnyT)),
        }
        try file.writeAll(&buf);
    }

    /// Helper for reading primitives with Little-Endian byte ordering.
    fn readAnyInt(file: std.fs.File, comptime AnyT: type) !AnyT {
        var buf: [@sizeOf(AnyT)]u8 = undefined;
        try readFull(file, &buf);
        switch (@typeInfo(AnyT)) {
            .int => return std.mem.readInt(AnyT, &buf, .little),
            .float => {
                const IntT = std.meta.Int(.unsigned, @sizeOf(AnyT) * 8);
                const int_val = std.mem.readInt(IntT, &buf, .little);
                return @bitCast(int_val);
            },
            else => unreachable,
        }
    }

    /// Recursive serialization logic for complex types and memory layouts.
    fn writeValue(file: std.fs.File, data: anytype) anyerror!void {
        const T = @TypeOf(data);
        switch (@typeInfo(T)) {
            .int, .float => try writeAnyInt(file, T, data),
            .bool => try file.writeAll(&[_]u8{if (data) 1 else 0}),
            .pointer => |info| {
                if (info.size == .slice) {
                    // Store slice length metadata
                    try writeAnyInt(file, u64, @as(u64, @intCast(data.len)));

                    // Alignment Padding: Ensure 4-byte boundaries for efficient SIMD/Hardware access
                    const pos = try file.getPos();
                    const padding = (4 - (pos % 4)) % 4;
                    if (padding > 0) {
                        try file.writeAll(("\x00" ** 3)[0..padding]);
                    }

                    // Bulk write binary payload
                    try file.writeAll(std.mem.sliceAsBytes(data));
                } else {
                    // Follow pointers for deep serialization
                    try writeValue(file, data.*);
                }
            },
            .@"struct" => |info| {
                // Compile-time reflection to iterate over struct fields
                inline for (info.fields) |field| {
                    try writeValue(file, @field(data, field.name));
                }
            },
            .array => try file.writeAll(std.mem.sliceAsBytes(&data)),
            else => @compileError("Unsupported type for serialization: " ++ @typeName(T)),
        }
    }

    /// Recursively reconstructs a data structure by reading from a file stream.
    /// Manages heap allocation for slices and pointers during the deserialization process.
    fn readValue(allocator: std.mem.Allocator, file: std.fs.File, comptime T: type) anyerror!T {
        switch (@typeInfo(T)) {
            .int, .float => return try readAnyInt(file, T),
            .bool => {
                var b: [1]u8 = undefined;
                try readFull(file, &b);
                return b[0] != 0;
            },
            .pointer => |info| {
                if (info.size == .slice) {
                    // Extract slice length from metadata
                    const len = try readAnyInt(file, u64);

                    // Alignment restoration: Jump to the next 4-byte boundary to synchronize with the serialized padding.
                    const pos = try file.getPos();
                    const padding = (4 - (pos % 4)) % 4;
                    if (padding > 0) try file.seekBy(@intCast(padding));

                    // Dynamic memory allocation for the slice payload
                    const slice = try allocator.alloc(info.child, @intCast(len));
                    try readFull(file, std.mem.sliceAsBytes(slice));
                    return slice;
                } else {
                    // Recursive allocation and reconstruction for single pointers
                    const res = try allocator.create(info.child);
                    res.* = try readValue(allocator, file, info.child);
                    return res;
                }
            },
            .@"struct" => |info| {
                var result: T = undefined;
                // Compile-time field iteration for structural reconstruction
                inline for (info.fields) |field| {
                    @field(result, field.name) = try readValue(allocator, file, field.type);
                }
                return result;
            },
            .array => {
                var result: T = undefined;
                // Bulk read for fixed-size arrays
                try readFull(file, std.mem.sliceAsBytes(&result));
                return result;
            },
            else => unreachable,
        }
    }

    /// Robust wrapper around the file read syscall to ensure the entire buffer is populated.
    /// Handles partial reads and prevents premature EndOfStream errors.
    fn readFull(file: std.fs.File, buffer: []u8) !void {
        var total_read: usize = 0;
        while (total_read < buffer.len) {
            const n = try file.read(buffer[total_read..]);
            if (n == 0) return error.EndOfStream;
            total_read += n;
        }
    }

    /// Maps the binary file directly into the virtual address space (Zero-Copy Deserialization).
    /// Optimized for large-scale weight buffers, reducing memory overhead and startup latency.
    /// Highly effective in resource-constrained environments like embedded systems or edge devices.
    pub fn loadMmap(filename: []const u8, comptime T: type) !T {
        var file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        // Header Validation (Magic Number)
        var magic_buf: [4]u8 = undefined;
        try readFull(file, &magic_buf);
        if (!std.mem.eql(u8, &magic_buf, GLOBAL_MAGIC)) return error.InvalidMagicNumber;

        // Skip metadata segment to locate the data offset
        _ = try readAnyInt(file, u32); // Version
        _ = try readAnyInt(file, u32); // Type size
        _ = try readAnyInt(file, u32); // Field count
        const name_len = try readAnyInt(file, u32);
        try file.seekBy(@intCast(name_len));

        const data_offset = try file.getPos();
        const file_size = (try file.stat()).size;

        // Establish memory mapping using POSIX mmap
        const ptr = try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );

        var offset: usize = data_offset;
        // Reconstruct the data structure using pointers into the mapped buffer
        return try readValueFromBuffer(ptr, &offset, T);
    }

    /// Deserializes data directly from a memory buffer without additional allocations.
    /// Leverages @alignCast to ensure hardware-level alignment requirements are met for sliced data.
    fn readValueFromBuffer(buffer: []u8, offset: *usize, comptime T: type) anyerror!T {
        switch (@typeInfo(T)) {
            .int, .float => {
                const size = @sizeOf(T);
                const IntT = std.meta.Int(.unsigned, size * 8);
                const val = std.mem.readInt(IntT, buffer[offset.*..][0..size], .little);
                offset.* += size;
                return @bitCast(val);
            },
            .bool => {
                const b = buffer[offset.*] != 0;
                offset.* += 1;
                return b;
            },
            .pointer => |info| {
                if (info.size == .slice) {
                    const len = std.mem.readInt(u64, buffer[offset.*..][0..8], .little);
                    offset.* += 8;

                    // Pointer Alignment: Bitwise rounding to the next 4-byte boundary.
                    // This satisfies the alignment requirements for @alignCast.
                    offset.* = (offset.* + 3) & ~@as(usize, 3);

                    const total_bytes = len * @sizeOf(info.child);
                    const raw_bytes = buffer[offset.*..][0..total_bytes];

                    // Zero-copy conversion from byte buffer to typed slice
                    const slice = std.mem.bytesAsSlice(info.child, raw_bytes);
                    offset.* += total_bytes;
                    return @alignCast(slice);
                }
                return error.NotSupportedForMmap;
            },
            .@"struct" => |info| {
                var result: T = undefined;
                inline for (info.fields) |field| {
                    @field(result, field.name) = try readValueFromBuffer(buffer, offset, field.type);
                }
                return result;
            },
            .array => {
                const size = @sizeOf(T);
                // Directly cast the buffer segment to the array value
                const result = std.mem.bytesAsValue(T, buffer[offset.*..][0..size]).*;
                offset.* += size;
                return result;
            },
            else => unreachable,
        }
    }
};
