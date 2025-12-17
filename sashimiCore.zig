const std = @import("std");

const Fish = struct {
    length: f32,
    width: f32,
};

fn predictWeight(fish: Fish) f32 {
    return (fish.length * 0.5) + (fish.width * 0.2);
}

pub fn main() !void {
    const file = try std.fs.cwd().createFile("fishDB.txt", .{ .truncate = true });
    defer file.close();

    const fishes = [_]Fish{
        .{ .length = 10.5, .width = 3.2 },
        .{ .length = 20.0, .width = 5.5 },
        .{ .length = 15.2, .width = 4.0 },
    };
    var buf: [100]u8 = undefined;
    for (fishes, 0..) |f, i| {
        const weight = predictWeight(f);
        const line = try std.fmt.bufPrint(&buf, "No. {d} weight: {d:.2}kg\n", .{ i + 1, weight });
        std.debug.print("Saving No. {d} fish weight...\n", .{i + 1});
        try file.writeAll(line);
    }
    std.debug.print("Sashimi DB update complete!\n", .{});
}
