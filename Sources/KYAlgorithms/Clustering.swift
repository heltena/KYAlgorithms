//
//  Clustering.swift
//  KYAlgorithms
//
//  Created by Heliodoro Tejedor Navarro on 24/4/23.
//

import Foundation

public protocol FittedModel {
    func predictedValues() -> [Int]
    func predict(numValues: Int, data: [Float]) throws -> [Int]
}

public enum Clustering {
    case gaussianMixture(numClusters: Int, numDimensions: Int)
    case kMeans(numClusters: Int, numDimensions: Int)
    
    public func fit(numValues: Int, data: [Float], maxIterations: Int = 100, tolerance: Float = 1e-6, verbose: Bool = false) async throws -> FittedModel {
        switch self {
        case .gaussianMixture(let numClusters, let numDimensions):
            let gmm = try GaussianMixture(numClusters: numClusters, numDimensions: numDimensions)
            return try await gmm.fit(numValues: numValues, data: data, maxIterations: maxIterations, tolerance: tolerance, verbose: verbose)
        case .kMeans(let numClusters, let numDimensions):
            let kmeans = try KMeans(numClusters: numClusters, numDimensions: numDimensions)
            return try await kmeans.fit(numValues: numValues, data: data, maxIterations: maxIterations, tolerance: tolerance, verbose: verbose)
        }
    }
}
