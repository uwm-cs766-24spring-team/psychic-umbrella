idea: take samples of backgrounds with nobody in it. reason: this is practical because scenes will often have empty samples. next: model gaussian mixture model of background distribution, evaluate probability per pixel for each sample. next, use class idea of hill climbing to find number of anomalies

advantage: very fast, no need to train on large dataset; mixture model is flexible, can adapt to multiple scenarios (day, night, weather, etc)
