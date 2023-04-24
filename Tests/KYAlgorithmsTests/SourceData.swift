import Accelerate
import Foundation
import XCTest

enum SourceDataError: Error {
    case wrongDimension
}

struct DataFile: Codable {
    var values: [Float32]
    var assigned: [Int]
}

func loadData(from resource: String) throws -> DataFile {
    let decoder = JSONDecoder()
    let url = try XCTUnwrap(Bundle.module.url(forResource: resource, withExtension: "json"))
    let data = try Data(contentsOf: url)
    return try decoder.decode(DataFile.self, from: data)
}

func generateData(using centers: [[Float]], numValues: Int, radius: Float = 1.0) throws -> [Float] {
    let centerDimensions = Set(centers.map { $0.count })
    if centerDimensions.count != 1 {
        throw SourceDataError.wrongDimension
    }
    let numDimensions = centerDimensions.first!
    let each_center = numValues / centers.count
    let extra_center = numValues - each_center * centers.count

    var n: Int32 = Int32(numValues * numDimensions)
    var d: Int32 = 3 // 3 for Normal(0, 1)
    var seed = [Int32](unsafeUninitializedCapacity: 4) { buffer, initializedCount in
        for i in 0..<3 {
            buffer[i] = Int32(arc4random_uniform(4096))
        }
        buffer[3] = Int32(arc4random_uniform(2048) << 1 + 1)
        initializedCount = 4
    }
    
    return try .init(unsafeUninitializedCapacity: Int(n)) { buffer, count in
        slarnv_(&d, &seed, &n, buffer.baseAddress)
        var current = 0
        for (index, center) in centers.enumerated() {
            try Task.checkCancellation()
            let size = index < centers.count - 1 ? each_center : each_center + extra_center
            for _ in 0..<size {
                for value in center {
                    buffer[current] *= radius
                    buffer[current] += value
                    current += 1
                }
            }
        }
        count = Int(n)
    }
}
