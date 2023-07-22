ndk-build
adb push ../obj/local/arm64-v8a/* /data/local/tmp
adb shell "export LD_LIBRARY_PATH=/data/local/tmp:${LD_LIBRARY_PATH} && cd /data/local/tmp && ./snpe-sample -h"
echo -e "\n"
adb shell "export LD_LIBRARY_PATH=/data/local/tmp:${LD_LIBRARY_PATH} && cd /data/local/tmp && ./snpe-sample -d inception_v3_quantized.dlc -i target_raw_list.txt -o ./out -l cpu,gpu,dsp"
