//
//  Clustering.swift
//  KYAlgorithms
//
//  Created by Heliodoro Tejedor Navarro on 24/4/23.
//

import Foundation

public protocol FittedModel {
    func predictedValues() -> [Int]
    func predict(numValues: Int, data: [Float]) -> [Int]
}

public enum Clustering {
    case gaussianMixture(numClusters: Int, numDimensions: Int)
    case kMeans(numClusters: Int, numDimensions: Int)
    
    public func fit(numValues: Int, data: [Float], maxIterations: Int = 100, tolerance: Float = 1e-6, verbose: Bool = false) -> FittedModel {
        switch self {
        case .gaussianMixture(let numClusters, let numDimensions):
            let gmm = GaussianMixture(numClusters: numClusters, numDimensions: numDimensions)
            return gmm.fit(numValues: numValues, data: data, maxIterations: maxIterations, tolerance: tolerance, verbose: verbose)
        case .kMeans(let numClusters, let numDimensions):
            let kmeans = KMeans(numClusters: numClusters, numDimensions: numDimensions)
            return kmeans.fit(numValues: numValues, data: data, maxIterations: maxIterations, tolerance: tolerance, verbose: verbose)
        }
    }
}
