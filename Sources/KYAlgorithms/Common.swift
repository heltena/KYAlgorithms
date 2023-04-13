//
//  Common.swift
//  KYAlgorithms
//
//  Created by Heliodoro Tejedor Navarro on 11/4/23.
//

import Accelerate
import Foundation

extension Collection where Index: Numeric {
    public func randomElement(usingWeights weights: [Float], weightSum: Float = 1.0) -> Element? {
        guard !isEmpty else { return nil }
        let value = Float.random(in: 0...1) * weightSum
        var accum: Float = 0.0
        for (weight, element) in zip(weights, self) {
            accum += weight
            if value <= accum {
                return element
            }
        }
        return self[Self.Index(exactly: vDSP.indexOfMaximum(weights).0) ?? startIndex]
    }
}
