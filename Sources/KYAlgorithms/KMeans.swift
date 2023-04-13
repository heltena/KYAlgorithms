//
//  KMeans.swift
//  KYAlgorithms
//
//  Created by Heliodoro Tejedor Navarro on 11/4/23.
//

import Accelerate
import Foundation

public struct KMeans {
    public let numClusters: Int
    public let numDimensions: Int

    public struct FitResult {
        let centroids: [[Float]]
        let numIterations: Int
        let inertia: Float
    }

    public init(numClusters: Int, numDimensions: Int) {
        self.numClusters = numClusters
        self.numDimensions = numDimensions
    }
    
    public func fit(numValues: Int, data: [Float], maxIterations: Int = 300, tolerance: Float = 1e-4, verbose: Bool = false) -> FitResult {
        let adaptedTolerance = dataVariance(numValues: numValues, data: data) * tolerance
        var assignedValues = [Int](unsafeUninitializedCapacity: numValues) { buffer, initializedCount in
            initializedCount = numValues
        }
        var centroids = kmeansPlusPlus(numValues: numValues, data: data)
        for currentIteration in 0..<maxIterations {
            assignValues(numValues: numValues, data: data, centroids: centroids, assignedValues: &assignedValues)
            let newCentroids = recalculateCentroids(numValues: numValues, data: data, centroids: centroids, assignations: assignedValues)
            if centroids == newCentroids {
                let inertia = calculateInertia(numValues: numValues, data: data, centroids: centroids, assignedValues: assignedValues)
                return .init(centroids: newCentroids, numIterations: currentIteration, inertia: inertia)
            }
            let error = calculateError(previousCentroids: centroids, centroids: newCentroids)
            if error < adaptedTolerance {
                let inertia = calculateInertia(numValues: numValues, data: data, centroids: newCentroids, assignedValues: assignedValues)
                return .init(centroids: newCentroids, numIterations: currentIteration, inertia: inertia)
            }
            centroids = newCentroids
            if verbose {
                let inertia = calculateInertia(numValues: numValues, data: data, centroids: centroids, assignedValues: assignedValues)
                print("Iteration \(currentIteration) inertia: \(inertia)")
            }
        }
        let inertia = calculateInertia(numValues: numValues, data: data, centroids: centroids, assignedValues: assignedValues)
        return .init(centroids: centroids, numIterations: maxIterations, inertia: inertia)
    }

    public func dataVariance(numValues: Int, data: [Float]) -> Float {
        var variances: [Float] = []
        for i in 0..<numDimensions {
            var mn: Float = 0.0
            var sddev: Float = 0.0
            var variance: Float = 0.0
            data.withUnsafeBytes { ptr in
                let x = ptr.bindMemory(to: Float.self)
                vDSP_normalize(x.baseAddress! + i, numDimensions, nil, 1, &mn, &sddev, vDSP_Length(numValues))
                variance = sddev * sddev
                variances.append(variance)
            }
        }
        return vDSP.mean(variances)
    }
    
    public func kmeansPlusPlus(numValues: Int, data: [Float]) -> [[Float]] {
        .init(unsafeUninitializedCapacity: numClusters) { buffer, initializedCount in
            var weights = Array(repeating: 1.0 / Float(numValues), count: numValues)

            var forbiddenIndices = Set<Int>([])
            for i in 0..<numClusters-1 {
                let newIndex = (0..<numValues).randomElement(usingWeights: weights, weightSum: vDSP.sum(weights))!
                let centroid = Array(data[newIndex * numDimensions..<(newIndex + 1) * numDimensions])
                buffer[i] = centroid
                forbiddenIndices.insert(newIndex)
                calculateWeights(numValues: numValues, data: data, centroid: centroid, weights: &weights)
                for forbiddenIndex in forbiddenIndices {
                    weights[forbiddenIndex] = 0.0
                }
            }
            let newIndex = (0..<numValues).randomElement(usingWeights: weights, weightSum: vDSP.sum(weights))!
            let centroid = Array(data[newIndex * numDimensions..<(newIndex + 1) * numDimensions])
            buffer[numClusters-1] = centroid
            initializedCount = numClusters
        }
    }

    public func calculateWeights(numValues: Int, data: [Float], centroid: [Float], weights: inout [Float]) {
        for index in 0..<numValues {
            let offset = index * numDimensions
            weights[index] = vDSP.distanceSquared(data[offset..<offset + numDimensions], centroid)
        }
    }
    
    public func assignValues(numValues: Int, data: [Float], centroids: [[Float]], assignedValues: inout [Int]) {
        let flattenCentroids = centroids.flatMap { $0 }
        let squaredCentroidSum: [Float] = centroids.map { vDSP.square($0).reduce(0, +) }
        let currentNumClusters = centroids.count // This function is uses at kmeansPlusPlus and need to partial assign the values
        
        // Init matrix as: M[i,j] = sum of the squared components of centroid[j]
        var matrix = Array<Float>(unsafeUninitializedCapacity: numValues * currentNumClusters) { buffer, initializedCount in
            for i in 0..<numValues {
                memcpy(buffer.baseAddress?.advanced(by: i * currentNumClusters), squaredCentroidSum, MemoryLayout<Float>.size * currentNumClusters)
            }
            initializedCount = numValues * currentNumClusters
        }
        
        // M[i,j] = M[i,j] - 2 * the sum of each dimension(d) { value[i][d] * centroid[j][d] }
        cblas_sgemm(
            CblasRowMajor,               // C Style
            CblasNoTrans,                // Data is transpose
            CblasTrans,                  // Centroids matrix is not transposed
            Int32(numValues),            // M (rows in A & C)
            Int32(currentNumClusters),   // N (columns in B & C)
            Int32(numDimensions),        // K (columns in A, rows in B)
            -2.0,                        // alpha
            data,                        // A
            Int32(numDimensions),        // The size of the first dimension of matrix A
            flattenCentroids,            // B
            Int32(numDimensions),        // The size of the first dimension of matrix B
            1.0,                         // beta
            &matrix,                     // C
            Int32(currentNumClusters))   // The size of the first dimension of matrix C

        // M[i,j] becomes the distance between value[i] and centroid[j], subtracting the value[i][d]^2, which is not needed to
        // find the closest:
        // distance(V, C) = sqrt( (v.x - c.x)^2 + (v.y - c.y)^2 ...) ~ (v.x - c.x)^2 + (v.y - c.y)^2...
        //                  (v.x - c.x)^2 = v.x^2 + c.x^2 - 2 * v.x * c.x ~ c.x^2 - 2 * v.x * c.x  (v.x^2 is a constant for all
        //                                                                                          the centroids)
        for i in 0..<numValues {
            assignedValues[i] = Int(vDSP.indexOfMinimum(matrix[i * currentNumClusters..<(i + 1) * currentNumClusters]).0)
        }
    }

    public func recalculateCentroids(numValues: Int, data: [Float], centroids: [[Float]], assignations: [Int]) -> [[Float]] {
        // Could not find a vectorized operation to find the mean of the values using a mask
        let numProcessors = ProcessInfo.processInfo.activeProcessorCount
        let numItemInChunks = numValues / numProcessors
        let numChunks = numValues.isMultiple(of: numItemInChunks) ? numValues / numItemInChunks : (numValues / numItemInChunks) + 1
        let remain = numValues.isMultiple(of: numItemInChunks) ? 0 : numValues - (numChunks - 1) * numItemInChunks
        var sums = Array(repeating: Float(0), count: numChunks * numClusters)
        let partialSums = [Float](unsafeUninitializedCapacity: numChunks * numClusters * numDimensions) { buffer, initializedCount in
            DispatchQueue.concurrentPerform(iterations: numChunks) { iteration in
                let size = iteration < numChunks - 1 ? numItemInChunks : remain
                var counts = Array(repeating: Int(0), count: numClusters)
                var chunkCounts = Array(repeating: Float(0), count: numClusters * numDimensions)
                for index in 0..<size {
                    let assignedValue = assignations[iteration * numItemInChunks + index]
                    for dimension in 0..<numDimensions {
                        chunkCounts[assignedValue * numDimensions + dimension] += data[(iteration * numItemInChunks + index) * numDimensions + dimension]
                    }
                    counts[assignedValue] += 1
                }
                for i in 0..<numClusters {
                    sums[i * numChunks + iteration] = Float(counts[i])
                }
                for i in 0..<numClusters * numDimensions {
                    buffer[i * numChunks + iteration] = chunkCounts[i]
                }
            }
            initializedCount = numChunks * numClusters * numDimensions
        }
        
        var guessCentroids: [[Float]] = []
        for centroidIndex in 0..<numClusters {
            let value = vDSP.sum(sums[centroidIndex * numChunks..<(centroidIndex + 1) * numChunks])
            let newCentroid = (0..<numDimensions).map { vDSP.sum(partialSums[(centroidIndex * numDimensions + $0) * numChunks..<(centroidIndex * numDimensions + $0 + 1) * numChunks]) / value }
            guessCentroids.append(newCentroid)
        }
        return guessCentroids
    }

    public func calculateError(previousCentroids: [[Float]], centroids: [[Float]]) -> Float {
        zip(previousCentroids, centroids)
            .map { vDSP.distanceSquared($0, $1) }
            .reduce(0, +)
    }

    public func calculateInertia(numValues: Int, data: [Float], centroids: [[Float]], assignedValues: [Int]) -> Float {
        var accum: Float = 0
        for (index, assignedValue) in assignedValues.enumerated() {
            let distance = vDSP.distanceSquared(data[index * numDimensions..<(index + 1) * numDimensions], centroids[assignedValue])
            accum += distance
        }
        return accum
    }
}
