idea: take samples of backgrounds with nobody in it. reason: this is practical because scenes will often have empty samples. next: model gaussian mixture model of background distribution, evaluate probability per pixel for each sample. next, use class idea of connected components to find number of anomalies

advantage: simple; mixture model is flexible, can adapt to multiple scenarios (day, night, weather, etc) for single scene, in practice makes sense
disadvantage: very slow, subject to tuning --> overfits to specific scene
