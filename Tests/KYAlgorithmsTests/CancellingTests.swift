import XCTest
@testable import KYAlgorithms


final class CancellingTests: XCTestCase {
    @available(macOS 13.0, *)
    func testPerformanceUsingDataFull() throws {
        let centers: [[Float]] = [
            [   0,    0,   0],
            [ 600,  600,   0],
            [1200,    0, 600]
        ]
        let numClusters = centers.count
        let numDimensions = centers[0].count
        let numValues = 10_000_000

        let exp = expectation(description: "Finished")
        let gmmTask = Task {
            do {
                print("Generating data...")
                let values = try generateData(using: centers, numValues: numValues, radius: 100.0)
                print("GaussianMixture...")
                let gmm = try GaussianMixture(numClusters: numClusters, numDimensions: numDimensions)
                let result = try await gmm.fit(numValues: numValues, data: values,  maxIterations: 400, tolerance: 1e-4)
                let sortedValues = zip(result.means, result.covariances).sorted(by: { lhs, rhs in lhs < rhs })
                let means = sortedValues.map { $0.0 }
                let covariances = sortedValues.map { $0.1 }
                print(means)
                print(covariances)
            } catch {
                print("Error catched!")
            }
            exp.fulfill()
        }
        
        Task {
            try await Task.sleep(for: .seconds(2.0)) // Wait 2 seconds and then...
            gmmTask.cancel() // ... cancel
            print("Cancel!")
        }
        wait(for: [exp])
    }
}
