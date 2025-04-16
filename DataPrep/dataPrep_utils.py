def normalizeData(dataSet, globalMin,globalMax):
    marginMin = globalMin - (globalMin*.2)
    marginMax = globalMax + (globalMax*.2)
    return (dataSet-marginMin) / (marginMax-marginMin)
