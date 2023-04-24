import XCTest
@testable import KYAlgorithms

final class KMeansTests: XCTestCase {   
    func testPerformanceUsingDataFull() throws {
        let values = try loadData(from: "data_full").values
        let numClusters = 2
        let numDimensions = 2
        let numValues = values.count / numDimensions
        let kmeans = try KMeans(numClusters: numClusters, numDimensions: numDimensions)
        
        measureMetrics([.wallClockTime], automaticallyStartMeasuring: false) {
            let exp = expectation(description: "Finished")
            Task {
                startMeasuring()
                let result = try await kmeans.fit(numValues: numValues, data: values,  maxIterations: 400, tolerance: 1e-4)
                stopMeasuring()
                print(result.centroids.sorted(by: {$0[0] < $1[0] }))
                exp.fulfill()
            }
            wait(for: [exp])
        }
    }
    
    func testPerformanceUsingNormalDistribution() throws {
        let centers: [[Float]] = [
            [   0,    0,   0],
            [ 600,  600,   0],
            [1200,    0, 600]
        ]
        let numClusters = centers.count
        let numDimensions = centers[0].count
        let numValues = 1_000_000

        let values = try generateData(using: centers, numValues: numValues, radius: 100.0)
        let kmeans = try KMeans(numClusters: numClusters, numDimensions: numDimensions)

        measureMetrics([.wallClockTime], automaticallyStartMeasuring: false) {
            let exp = expectation(description: "Finished")
            Task {
                startMeasuring()
                let result = try await kmeans.fit(numValues: numValues, data: values,  maxIterations: 400, tolerance: 1e-4)
                stopMeasuring()
                print(result.centroids.sorted(by: {$0[0] < $1[0] }))
                exp.fulfill()
            }
            wait(for: [exp])
        }
    }
}
