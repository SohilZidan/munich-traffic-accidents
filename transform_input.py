#!/usr/bin/env python3

def transform(sample_raw):
    sample = [
        int(sample_raw[2]),
        int(sample_raw[3]),
        int(sample_raw[0] == "Alkoholunfälle"),
        int(sample_raw[0] == "Fluchtunfälle"),
        int(sample_raw[0] == "Verkehrsunfälle"),
        int(sample_raw[1] == "Verletzte und Getötete"),
        int(sample_raw[1] == "insgesamt"),
        int(sample_raw[1] == "mit Personenschäden")
    ]
    return sample
