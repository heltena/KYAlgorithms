import XCTest
@testable import KYAlgorithms

final class HistogramTests: XCTestCase {
    func testSimpleHistogram() throws {
        let values: [Float] = (0..<1000).map { $0.isMultiple(of: 2) ? 100 : 0 }
        let histogram = try Histogram(values: values, useLog10: false, binsCount: 10)
        XCTAssertTrue(histogram.data[0] == 500 && histogram.data[9] == 500)
    }
    
    func testMultipleHistogram() throws {
        let values: [Float] = (0..<1000).flatMap {
            [
                $0.isMultiple(of: 2) ? 100 : 0,
                $0.isMultiple(of: 3) ? 100 : 0,
                $0.isMultiple(of: 4) ? 100 : 0,
            ]
        }
        let histogram = try Histogram(values: values, orderedChannelNames: ["mul2", "mul3", "mul4"], useLog10: false, binsCount: 10)
        print(histogram.data)
        XCTAssertTrue(histogram.data[0] == 500 && histogram.data[27] == 500)
        XCTAssertTrue(histogram.data[1] == 666 && histogram.data[28] == 334)
        XCTAssertTrue(histogram.data[2] == 750 && histogram.data[29] == 250)
    }
    
    func testMultipleHistogramPerformance() throws {
        let values: [Float] = (0..<100_000).flatMap {
            [
                $0.isMultiple(of: 2) ? 100 : 0,
                $0.isMultiple(of: 3) ? 100 : 0,
                $0.isMultiple(of: 4) ? 100 : 0,
            ]
        }
        measure {
            do {
                let histogram = try Histogram(values: values, orderedChannelNames: ["mul2", "mul3", "mul4"], useLog10: false, binsCount: 10)
                XCTAssertTrue(histogram.data[0] == 50000 && histogram.data[27] == 50000)
                XCTAssertTrue(histogram.data[1] == 66666 && histogram.data[28] == 33334)
                XCTAssertTrue(histogram.data[2] == 75000 && histogram.data[29] == 25000)
            } catch {
                print("Error")
            }
        }
    }

}
