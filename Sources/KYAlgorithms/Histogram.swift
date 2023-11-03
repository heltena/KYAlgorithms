//
//  Histogram.swift
//  
//
//  Created by Heliodoro Tejedor Navarro on 19/8/23.
//

import Accelerate
import Foundation

public struct Histogram {
    public enum Kind {
        case raw
        case density
    }

    public enum HistogramError: Error {
        case invalidInputValues
    }

    public struct Element: Identifiable {
        public var id: Int
        public var x: Float
        public var frequency: Float
        
        public init(id: Int, x: Float, frequency: Float) {
            self.id = id
            self.x = x
            self.frequency = frequency
        }
    }
    
    public struct Channel: Identifiable {
        public var id: UUID
        public var name: String
        public var elements: [Element]
        public var minXValue: Float
        public var maxXValue: Float
        public var maxFrequency: Float
        public var xStep: Float
        
        public init(id: UUID, name: String, elements: [Element], minXValue: Float, maxXValue: Float, maxFrequency: Float, xStep: Float) {
            self.id = id
            self.name = name
            self.elements = elements
            self.minXValue = minXValue
            self.maxXValue = maxXValue
            self.maxFrequency = maxFrequency
            self.xStep = xStep
        }
    }

    public let kind: Kind
    public let usedLog10: Bool
    public let binsCount: Int
    public let valuesCount: Int
    public let channels: [Channel]
    
    public init(kind: Kind, usedLog10: Bool, binsCount: Int, valuesCount: Int, channels: [Channel]) {
        self.kind = kind
        self.usedLog10 = usedLog10
        self.binsCount = binsCount
        self.valuesCount = valuesCount
        self.channels = channels
    }
    
    public init(kind: Kind, values: [Float], useLog10: Bool, binsCount: Int) throws {
        var data = Array(repeating: 0, count: binsCount)
        let buffer = useLog10
            ? values.filter { $0 >= 1.0 }.map { log10($0) }
            : values
        let minValue = vDSP.minimum(buffer)
        let maxValue = vDSP.maximum(buffer)
        let step = (maxValue - minValue) / Float(binsCount)
        if step == 0 {
            data[0] = values.count
        } else {
            for value in buffer {
                let index = min(binsCount-1, Int((value - minValue) / step))
                data[index] += 1
            }
        }
        let maxFrequency = data.max() ?? 0
        
        let factor: Float
        switch kind {
        case .raw:
            factor = 1.0
        case .density:
            factor = (maxValue - minValue) * step / Float(values.count)
        }
        let elements = data.enumerated().map { offset, element in
            Element(id: offset, x: step * (Float(offset) + 0.5), frequency: Float(element) * factor)
        }
        
        self.kind = kind
        self.usedLog10 = useLog10
        self.binsCount = binsCount
        self.valuesCount = values.count
        self.channels = [Channel(id: UUID(), name: "", elements: elements, minXValue: minValue, maxXValue: maxValue, maxFrequency: Float(maxFrequency), xStep: step)]
    }
    
    /// values contains the channel values interlined:
    ///   Ch0V0 Ch1V0 Ch2V0  Ch0V1 Ch1V1 Ch2V1  Ch0V2 Ch2V2 Ch3V2  ...
    public init(kind: Kind, values: [Float], orderedChannelNames: [String], useLog10: Bool, binsCount: Int) throws {
        guard
            binsCount > 0,
            values.count >= 2 * orderedChannelNames.count,
            values.count % orderedChannelNames.count == 0
        else {
            throw HistogramError.invalidInputValues
        }
        
        var channels: [Channel] = []
        var data = Array(repeating: 0, count: binsCount * orderedChannelNames.count)
        let valuesCount = values.count / orderedChannelNames.count
        for (channelIndex, channelName) in orderedChannelNames.enumerated() {
            try Task.checkCancellation()
            let buffer: [Float]
            if useLog10 {
                buffer = (0..<valuesCount)
                    .map { values[$0 * orderedChannelNames.count + channelIndex] }
                    .filter { $0 >= 1.0 }
                    .map { log10($0) }
            } else {
                buffer = (0..<valuesCount)
                    .map { values[$0 * orderedChannelNames.count + channelIndex] }
            }
            
            let minValue = vDSP.minimum(buffer)
            let maxValue = vDSP.maximum(buffer)
            let step = (maxValue - minValue) / Float(binsCount)
            if step == 0 {
                data[0 * orderedChannelNames.count + channelIndex] = values.count
            } else {
                for value in buffer where value.isFinite {
                    let index = min(binsCount - 1, Int((value - minValue) / step))
                    data[index * orderedChannelNames.count + channelIndex] += 1
                }
            }
            let histogramValues = (0..<binsCount)
                .map { data[$0 * orderedChannelNames.count + channelIndex] }
            
            let maxFrequency = histogramValues.max() ?? 0
            
            let factor: Float
            switch kind {
            case .raw:
                factor = 1.0
            case .density:
                factor = (maxValue - minValue) * step / Float(valuesCount)
            }
            
            let elements = histogramValues.enumerated().map { offset, element in
                Element(id: offset, x: step * (Float(offset) + 0.5), frequency: Float(element) * factor)
            }
            
            let channel = Channel(id: UUID(), name: channelName, elements: elements, minXValue: minValue, maxXValue: maxValue, maxFrequency: Float(maxFrequency), xStep: step)
            channels.append(channel)
        }

        self.kind = kind
        self.usedLog10 = useLog10
        self.binsCount = binsCount
        self.valuesCount = valuesCount
        self.channels = channels
    }
}
