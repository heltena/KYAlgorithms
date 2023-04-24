//
//  GaussianMixture.swift
//  KYAlgorithms
//
//  Created by Heliodoro Tejedor Navarro on 24/4/23.
//

import Accelerate
import Foundation

// This code is based on the scikit-learn implementation: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
//
// The authors as shown in the source code:
// Author: Wei Xue <xuewei4d@gmail.com>
// Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
// License: BSD 3 clause


public struct GaussianMixture {
    public let numClusters: Int
    public let numDimensions: Int
    
    struct GaussianParameters {
        var weights: [Float]
        var means: [Float]
        var covariances: [Float]
        var precisionsCholesky: [Float]
    }

    public struct FitResult: FittedModel {
        let gmm: GaussianMixture
        public let converged: Bool
        public let lowerBound: Float
        public let numIterations: Int
        private let parameters: GaussianParameters
        let assignedValues: [Int]
        
        public var weights: [Float] { parameters.weights }
        public var means: [Float] { parameters.means }
        public var covariances: [Float] { parameters.covariances }
        public var precisionsCholesky: [Float] { parameters.precisionsCholesky }
        
        init(gmm: GaussianMixture, converged: Bool, lowerBound: Float, numIterations: Int, parameters: GaussianParameters, assignedValues: [Int]) {
            self.gmm = gmm
            self.converged = converged
            self.lowerBound = lowerBound
            self.numIterations = numIterations
            self.parameters = parameters
            self.assignedValues = assignedValues
        }
        
        public func predictedValues() -> [Int] {
            assignedValues
        }
        
        public func predict(numValues: Int, data: [Float]) -> [Int] {
            let squaredData = vDSP.square(data)
            let (_, logResp) = gmm.eStep(numValues: numValues, data: data, squaredData: squaredData, parameters: parameters)
            return (0..<numValues).map {
                Int(vDSP.indexOfMaximum(logResp[$0 * gmm.numClusters..<($0 + 1) * gmm.numClusters]).0)
            }
        }
    }

    public init(numClusters: Int, numDimensions: Int) {
        self.numClusters = numClusters
        self.numDimensions = numDimensions
    }
   
    public func fit(numValues: Int, data: [Float], repetitions: Int = 1, maxIterations: Int = 100, tolerance: Float = 1e-3, verbose: Bool = false) -> FitResult {
        var bestResult: FitResult?
        let squaredData = vDSP.square(data)
        for _ in 0..<repetitions {
            let result = singleFit(numValues: numValues, data: data, squaredData: squaredData, maxIterations: maxIterations, tolerance: tolerance, verbose: verbose)
            if let previousResult = bestResult {
                if previousResult.lowerBound < result.lowerBound {
                    bestResult = result
                }
            } else {
                bestResult = result
            }
        }
        return bestResult!
    }

    func singleFit(numValues: Int, data: [Float], squaredData: [Float], maxIterations: Int = 100, tolerance: Float = 1e-3, regCovar: Float = 1e-6, verbose: Bool = false) -> FitResult {
        let kmeans = KMeans(numClusters: numClusters, numDimensions: numDimensions)
        let initialFit = kmeans.fit(numValues: numValues, data: data)
        var resp = Array(repeating: Float(0), count: numValues * numClusters)
        for (value, cluster) in initialFit.assignedValues.enumerated() {
            resp[value * numClusters + cluster] = 1.0
        }

        var parameters = estimateGaussianParameters(numValues: numValues, data: data, resp: resp, regCovar: regCovar)

        var converged = false
        var lowerBound = -Float.infinity
        var iteration = 0
        while !converged && iteration < maxIterations {
            let prevLowerBound = lowerBound
            let (logProbNorm, logResp) = eStep(numValues: numValues, data: data, squaredData: squaredData, parameters: parameters)
            
            // mStep
            let expLogResp = logResp.map { exp($0) }
            parameters = estimateGaussianParameters(numValues: numValues, data: data, resp: expLogResp, regCovar: regCovar)
            lowerBound = logProbNorm
            
            let change = lowerBound - prevLowerBound
            if abs(change) < tolerance {
                converged = true
                break
            }
            iteration += 1
        }
        
        let (_, logResp) = eStep(numValues: numValues, data: data, squaredData: squaredData, parameters: parameters)
        let assignedValues = (0..<numValues).map {
            Int(vDSP.indexOfMaximum(logResp[$0 * numClusters..<($0 + 1) * numClusters]).0)
        }
        return .init(gmm: self, converged: converged, lowerBound: lowerBound, numIterations: iteration, parameters: parameters, assignedValues: assignedValues)
    }
    
    func estimateGaussianParameters(numValues: Int, data: [Float], resp: [Float], regCovar: Float) -> GaussianParameters {
        // Calculate weights
        let weights = Array<Float>(unsafeUninitializedCapacity: numClusters, initializingWith: { buffer, initializedCount in
            resp.withUnsafeBufferPointer { ptr in
                for cluster in 0..<numClusters {
                    var result: Float = 0
                    vDSP_sve(
                        ptr.baseAddress!.advanced(by: cluster),
                        vDSP_Stride(numClusters),
                        &result,
                        vDSP_Length(resp.count / numClusters))
                    buffer[cluster] = result + Float.leastNonzeroMagnitude * 10 // make sure is not zero
                }
            }
            initializedCount = numClusters
        })

        // Calculate means
        var means = Array(repeating: Float(0), count: numDimensions * numClusters)
        
        // means = resp.transpose x data
        cblas_sgemm(
            CblasRowMajor,        // C Style
            CblasTrans,           // resp is transpose
            CblasNoTrans,         // data matrix is not transposed
            Int32(numClusters),   // M (rows in A & C)
            Int32(numDimensions), // N (columns in B & C)
            Int32(numValues),     // K (columns in A, rows in B)
            1.0,                  // alpha
            resp,                 // A
            Int32(numClusters),   // The size of the first dimension of matrix A
            data,                 // B
            Int32(numDimensions), // The size of the first dimension of matrix B
            1.0,                  // beta
            &means,               // C
            Int32(numDimensions)) // The size of the first dimension of matrix C
        
        // means /= weights
        means.withUnsafeMutableBufferPointer { ptr in
            for (index, var weight) in weights.enumerated() {
                vDSP_vsdiv(
                    ptr.baseAddress!.advanced(by: index * numDimensions),
                    vDSP_Stride(1),
                    &weight,
                    ptr.baseAddress!.advanced(by: index * numDimensions),
                    vDSP_Stride(1),
                    vDSP_Length(numDimensions))
            }
        }

        // covariances (diag) = avg(X)^2 - 2 * avg(X) * means + means^2 + reg_covar
        let data2 = vDSP.square(data)
        let means2 = vDSP.square(means)
        
        var avgData2 = Array(repeating: Float(0), count: numDimensions * numClusters)
        
        // avgData2 = resp.transpose x data ^ 2
        cblas_sgemm(
            CblasRowMajor,        // C Style
            CblasTrans,           // resp is transpose
            CblasNoTrans,         // data matrix is not transposed
            Int32(numClusters),   // M (rows in A & C)
            Int32(numDimensions), // N (columns in B & C)
            Int32(numValues),     // K (columns in A, rows in B)
            1.0,                  // alpha
            resp,                 // A
            Int32(numClusters),   // The size of the first dimension of matrix A
            data2,                // B
            Int32(numDimensions), // The size of the first dimension of matrix B
            1.0,                  // beta
            &avgData2,            // C
            Int32(numDimensions)) // The size of the first dimension of matrix C
        
        // avgData2 /= weights
        avgData2.withUnsafeMutableBufferPointer { ptr in
            for (index, var weight) in weights.enumerated() {
                vDSP_vsdiv(
                    ptr.baseAddress!.advanced(by: index * numDimensions),
                    vDSP_Stride(1),
                    &weight,
                    ptr.baseAddress!.advanced(by: index * numDimensions),
                    vDSP_Stride(1),
                    vDSP_Length(numDimensions))
            }
        }

        let covariances = vDSP.add(regCovar, vDSP.subtract(avgData2, means2))
        let precisionsCholesky = covariances.map { 1.0 / sqrt($0) }
        return .init(weights: weights, means: means, covariances: covariances, precisionsCholesky: precisionsCholesky)
    }
    
    func eStep(numValues: Int, data: [Float], squaredData: [Float], parameters: GaussianParameters) -> (logProbNormMean: Float, logResp: [Float]) {
        let logPrecisionsCholesky = parameters.precisionsCholesky.map { log($0) }
        let logDet = Array<Float>(unsafeUninitializedCapacity: numClusters) { buffer, initializedCount in
            logPrecisionsCholesky.withUnsafeBufferPointer { ptr in
                for cluster in 0..<numClusters {
                    vDSP_sve(
                        ptr.baseAddress!.advanced(by: cluster * numDimensions),
                        vDSP_Stride(1),
                        buffer.baseAddress!.advanced(by: cluster),
                        vDSP_Length(numDimensions))
                }
            }
            initializedCount = numClusters
        }
        let precisions = vDSP.square(parameters.precisionsCholesky)
        
        let meansPrecisions = vDSP.multiply(parameters.means, precisions)
        let means2 = vDSP.square(parameters.means)
        let means2Precisions = vDSP.multiply(means2, precisions)
        let means2PrecisionsSum = Array<Float>(unsafeUninitializedCapacity: numClusters) { buffer, initializedCount in
            means2Precisions.withUnsafeBufferPointer { ptr in
                for cluster in 0..<numClusters {
                    vDSP_sve(
                        ptr.baseAddress!.advanced(by: cluster * numDimensions),
                        vDSP_Stride(1),
                        buffer.baseAddress!.advanced(by: cluster),
                        vDSP_Length(numDimensions))
                }
            }
            initializedCount = numClusters
        }

        // logProb = for each value, the sum of means^2 * precisions for each cluster
        var logProb = Array<Float>(unsafeUninitializedCapacity: numValues * numClusters) { buffer, initializedCount in
            means2PrecisionsSum.withUnsafeBufferPointer { ptr in
                for value in 0..<numValues {
                    memcpy(buffer.baseAddress!.advanced(by: value * numClusters), ptr.baseAddress!, numClusters * MemoryLayout<Float>.size)
                }
            }
            initializedCount = numValues * numClusters
        }
                             
        // logProb = -2 * (X x (means * precisions).T) + logProb
        cblas_sgemm(
            CblasRowMajor,        // C Style
            CblasNoTrans,         // data is not transpose
            CblasTrans,           // means * precisions is transposed
            Int32(numValues),     // M (rows in A & C)
            Int32(numClusters),   // N (columns in B & C)
            Int32(numDimensions), // K (columns in A, rows in B)
            -2.0,                 // alpha
            data,                 // A
            Int32(numDimensions), // The size of the first dimension of matrix A
            meansPrecisions,      // B
            Int32(numDimensions), // The size of the first dimension of matrix B
            1.0,                  // beta
            &logProb,             // C
            Int32(numClusters))   // The size of the first dimension of matrix C

        // then logProb = X^2 x precisions.T + logProb
        cblas_sgemm(
            CblasRowMajor,        // C Style
            CblasNoTrans,         // squaredData is not transpose
            CblasTrans,           // precisions is transposed
            Int32(numValues),     // M (rows in A & C)
            Int32(numClusters),   // N (columns in B & C)
            Int32(numDimensions), // K (columns in A, rows in B)
            1.0,                  // alpha
            squaredData,          // A
            Int32(numDimensions), // The size of the first dimension of matrix A
            precisions,           // B
            Int32(numDimensions), // The size of the first dimension of matrix B
            1.0,                  // beta
            &logProb,             // C
            Int32(numClusters))   // The size of the first dimension of matrix C
        
        // weightedLogProb = -0.5 * (numClusters * log(2 * pi) + logProb) + logDet
        let weightedLogProb = Array<Float>(unsafeUninitializedCapacity: numValues * numClusters) { buffer, initializedCount in
            for i in 0..<numValues * numClusters {
                buffer[i] = -0.5 * (Float(numClusters) * log(2 * Float.pi) + logProb[i]) + logDet[i % numClusters] + log(parameters.weights[i % numClusters])
            }
            initializedCount = numValues * numClusters
        }
        
        let logProbNorm = Array<Float>(unsafeUninitializedCapacity: numValues) { buffer, initializedCount in
            for i in 0..<numValues {
                let sumExp = weightedLogProb[i * numClusters..<(i + 1) * numClusters]
                    .map { exp($0) }
                    .reduce(0, +)
                buffer[i] = log(sumExp)
            }
            initializedCount = numValues
        }
        
        let logProbNormMean = vDSP.mean(logProbNorm)

        let logResp = Array<Float>(unsafeUninitializedCapacity: numValues * numClusters) { buffer, initializedCount in
            for i in 0..<numValues*numClusters {
                buffer[i] = weightedLogProb[i] - logProbNorm[i / numClusters]
            }
            initializedCount = numValues * numClusters
        }
        return (logProbNormMean: logProbNormMean, logResp: logResp)
    }

}
