import CreateML
import XCTest
@testable import KYAlgorithms

final class GaussianMixtureTests: XCTestCase {
    func testPerformanceUsingDataFull() throws {
        let values = try loadData(from: "data_full").values
        let numClusters = 2
        let numDimensions = 2
        let numValues = values.count / numDimensions
        let gmm = try GaussianMixture(numClusters: numClusters, numDimensions: numDimensions)
        
        measureMetrics([.wallClockTime], automaticallyStartMeasuring: false) {
            let exp = expectation(description: "Finished")
            Task {
                startMeasuring()
                let result = try await gmm.fit(numValues: numValues, data: values,  maxIterations: 400, tolerance: 1e-4)
                stopMeasuring()
                let sortedValues = zip(result.means, result.covariances).sorted(by: { lhs, rhs in lhs < rhs })
                let means = sortedValues.map { $0.0 }
                let covariances = sortedValues.map { $0.1 }
                print(means)
                print(covariances)
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
        let gmm = try GaussianMixture(numClusters: numClusters, numDimensions: numDimensions)

        measureMetrics([.wallClockTime], automaticallyStartMeasuring: false) {
            let exp = expectation(description: "Finished")
            Task {
                startMeasuring()
                let result = try await gmm.fit(numValues: numValues, data: values,  maxIterations: 400, tolerance: 1e-4)
                stopMeasuring()
                let sortedValues = zip(result.means, result.covariances).sorted(by: { lhs, rhs in lhs < rhs })
                let means = sortedValues.map { $0.0 }
                let covariances = sortedValues.map { $0.1 }
                print(means)
                print(covariances)
                exp.fulfill()
            }
            wait(for: [exp])
        }
    }
    
    func testExampleGMM() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "gmm_values", withExtension: "csv"))
        let table = try MLDataTable(contentsOf: url)
        
        let numDimensions = 10
        let numClusters = 3
        let values = (0..<table.rows.count).compactMap { table["value"].doubles?.element(at: $0) }.map { Float($0) }
        let numValues = values.count / numDimensions
        
        let gmm = try GaussianMixture(numClusters: numClusters, numDimensions: numDimensions)
        let exp = expectation(description: "Finished")
        Task {
            let result = try await gmm.fit(numValues: numValues, data: values, repetitions: 1, maxIterations: 400, tolerance: 1e-4)
            print("Weights")
            print(result.weights)

            print("Means")
            for i in 0..<numClusters {
                print(result.means[i * numDimensions..<(i + 1) * numDimensions])
            }
            
            print("Covariances")
            for i in 0..<numClusters {
                print(result.covariances[i * numDimensions..<(i + 1) * numDimensions])
            }
            exp.fulfill()
        }
        wait(for: [exp])
    }

    func testExampleGMMKMeansSet() throws {
        struct LoadingData: Codable {
            var points: [Float]
            var centers: [Float]
            var labels: [Int]
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "to_gmm_kmeans_data", withExtension: "json"))
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder.Rainflow()
        let info = try decoder.decode(LoadingData.self, from: data)
        
        let numDimensions = 10
        let numClusters = 3
        let numValues = info.points.count / numDimensions

        let gmm = try GaussianMixture(numClusters: numClusters, numDimensions: numDimensions)
        let exp = expectation(description: "Finished")
        Task {
            var initialResp = Array(repeating: Float(0), count: numValues * numClusters)
            for (value, cluster) in info.labels.enumerated() {
                initialResp[value * numClusters + cluster] = 1.0
            }

            let result = try await gmm.fit(numValues: numValues, data: info.points, repetitions: 1, maxIterations: 400, tolerance: 1e-4, initialResp: initialResp)
            print("Weights")
            print(result.weights)

            print("Means")
            for i in 0..<numClusters {
                print(result.means[i * numDimensions..<(i + 1) * numDimensions])
            }
            
            print("Covariances")
            for i in 0..<numClusters {
                print(result.covariances[i * numDimensions..<(i + 1) * numDimensions])
            }
            exp.fulfill()
        }
        wait(for: [exp])
    }
}
