# SNPE_samplecode
SNPE SDK 里面的sample code 实现多线程

## 目标
1. 支持同时推理多个实例，并显示每个实例的FPS，以及总的FPS。
2. 支持导出单实例时， 打印每层所用的时间；
3. 支持总的测试，由CPU，GPU，DSP。一级一级判断，如果有的话就做推理，并将结果打印到文件。