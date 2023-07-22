//==============================================================================
//
//  Copyright (c) 2015-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>

#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#ifdef ENABLE_GL_BUFFER
#include <GLES2/gl2.h>
#endif

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "DlSystem/PlatformConfig.hpp"

#include "Util.hpp"
#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "LoadUDOPackage.hpp"
#include "SetBuilderOptions.hpp"
#include "PreprocessInput.hpp"
const int FAILURE = 1;
const int SUCCESS = 0;

int main(int argc, char** argv)
{
    enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16};
    enum {CPUBUFFER, GLBUFFER};

    // Command line arguments
    static std::string dlc = "";
    static std::string OutputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "ITENSOR";
    std::string userBufferSourceStr = "CPUBUFFER";
    std::string staticQuantizationStr = "false";
    static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    static zdl::DlSystem::RuntimeList runtimeList;
    bool runtimeSpecified = false;
    bool execStatus = false;
    bool usingInitCaching = false;
    bool staticQuantization = false;
    bool cpuFixedPointMode = false;
    std::string UdoPackagePath = "";

    #ifdef __ANDROID__
    // Initialize Logs with level LOG_ERROR.
    zdl::SNPE::SNPEFactory::initializeLogging(zdl::DlSystem::LogLevel_t::LOG_ERROR);
    #else
    // Initialize Logs with level LOG_ERROR and Log path "./Log"
    zdl::SNPE::SNPEFactory::initializeLogging(zdl::DlSystem::LogLevel_t::LOG_ERROR, "./Log");
    #endif

    // Update Log level to LOG_WARN
    zdl::SNPE::SNPEFactory::setLogLevel(zdl::DlSystem::LogLevel_t::LOG_WARN);

    // Process command line arguments
    int opt = 0;
    while ((opt = getopt(argc, argv, "hi:d:o:b:q:s:z:r:l:u:c:x")) != -1)
    {
        switch (opt)
        {
        case 'h':
               std::cout
                        << "\nDESCRIPTION:\n"
                        << "------------\n"
                        << "Example application demonstrating how to load and execute a neural network\n"
                        << "using the SNPE C++ API.\n"
                        << "\n\n"
                        << "REQUIRED ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -d  <FILE>   Path to the DL container containing the network.\n"
                        << "  -i  <FILE>   Path to a file listing the inputs for the network.\n"
                        << "  -o  <PATH>   Path to directory to store output results.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -b  <TYPE>   Type of buffers to use [USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16] (" << bufferTypeStr << " is default).\n"
                        << "  -q  <BOOL>    Specifies to use static quantization parameters from the model instead of input specific quantization [true, false]. Used in conjunction with USERBUFFER_TF8. \n"
                        << "  -r  <RUNTIME> The runtime to be used [gpu, dsp, aip, cpu] (cpu is default). \n"
                        << "  -u  <VAL,VAL> Path to UDO package with registration library for UDOs. \n"
                        << "                Optionally, user can provide multiple packages as a comma-separated list. \n"
                        << "  -z  <NUMBER>  The maximum number that resizable dimensions can grow into. \n"
                        << "                Used as a hint to create UserBuffers for models with dynamic sized outputs. Should be a positive integer and is not applicable when using ITensor. \n"
                #ifdef ENABLE_GL_BUFFER
                        << "  -s  <TYPE>   Source of user buffers to use [GLBUFFER, CPUBUFFER] (" << userBufferSourceStr << " is default).\n"
                #endif
                        << "  -c           Enable init caching to accelerate the initialization process of SNPE. Defaults to disable.\n"
                        << "  -l  <VAL,VAL,VAL> Specifies the order of precedence for runtime e.g  cpu_float32, dsp_fixed8_tf etc. Valid values are:- \n"
                        << "                    cpu_float32 (Snapdragon CPU)       = Data & Math: float 32bit \n"
                        << "                    gpu_float32_16_hybrid (Adreno GPU) = Data: float 16bit Math: float 32bit \n"
                        << "                    dsp_fixed8_tf (Hexagon DSP)        = Data & Math: 8bit fixed point Tensorflow style format \n"
                        << "                    gpu_float16 (Adreno GPU)           = Data: float 16bit Math: float 16bit \n"
                #if DNN_RUNTIME_HAVE_AIP_RUNTIME
                        << "                    aip_fixed8_tf (Snapdragon HTA+HVX) = Data & Math: 8bit fixed point Tensorflow style format \n"

                #endif
                        << "                    cpu (Snapdragon CPU)               = Same as cpu_float32 \n"
                        << "                    gpu (Adreno GPU)                   = Same as gpu_float32_16_hybrid \n"
                        << "                    dsp (Hexagon DSP)                  = Same as dsp_fixed8_tf \n"
                #if DNN_RUNTIME_HAVE_AIP_RUNTIME
                        << "                    aip (Snapdragon HTA+HVX)           = Same as aip_fixed8_tf \n"
                #endif
                        << "  -x            Specifies to use the fixed point execution on CPU runtime for quantized DLC.\n"
                        << "                Used in conjunction with CPU runtime.\n"
                        << std::endl;
                std::exit(SUCCESS);
        case 'i':
            inputFile = optarg;
            break;
        case 'd':
            dlc = optarg;
            break;
        case 'o':
            OutputDir = optarg;
            break;
        case 'b':
            bufferTypeStr = optarg;
            break;
        case 'q':
            staticQuantizationStr = optarg;
            break;
        case 's':
            userBufferSourceStr = optarg;
            break;
        case 'z':
            setResizableDim(atoi(optarg));
            break;
        case 'r':
            runtimeSpecified = true;
            if (strcmp(optarg, "gpu") == 0)
            {
                runtime = zdl::DlSystem::Runtime_t::GPU;
            }
            else if (strcmp(optarg, "aip") == 0)
            {
                runtime = zdl::DlSystem::Runtime_t::AIP_FIXED_TF;
            }
            else if (strcmp(optarg, "dsp"))
            {
                runtime = zdl::DlSystem::Runtime_t::DSP;
            }
            else if (strcmp(optarg, "cpu") == 0)
            {
                runtime = zdl::DlSystem::Runtime_t::CPU;
            }
            else
            {
                std::cerr << "The runtime option provide is not valid. Defaulting to the CPU runtime." << std::endl;
            }
            break;
        case 'l':
        {
            std::string inputString = optarg;
            std::cout << "Input String: " << inputString << std::endl;
            std::vector<std::string> runtimeStrVector;
            split(runtimeStrVector, inputString, ',');

            // Check for dups
            for(auto it = runtimeStrVector.begin(); it!=runtimeStrVector.end() -1; it++) {
                auto found = std::find(it+1, runtimeStrVector.end(), *it);
                if (found != runtimeStrVector.end())
                {
                    std::cerr << "Error: Invalid values passed to the argument " << argv[optind -2] << ". Duplicate entries in runtime order." << std::endl;
                    std::exit(FAILURE); 
                }
            }

            runtimeList.clear();
            for (auto &runtimeStr : runtimeStrVector)
            {
                zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::RuntimeList::stringToRuntime(runtimeStr.c_str());
                if (runtime != zdl::DlSystem::Runtime_t::UNSET)
                {
                    bool ret = runtimeList.add(runtime);
                    if (ret == false)
                    {
                        std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
                        std::cerr << "Error: Invalid values passed to the argument " << argv[optind - 2] << ". Please provide comma seperated runtime order of precedence." << std::endl;
                        std::exit(FAILURE);
                    }
                }
                else
                {
                        std::cerr << "Error: Invalid values passed to the argument "<< argv[optind-2] << ". Please provide comma seperated runtime order of precedence" << std::endl;
                        std::exit(FAILURE);
                }
            }
        }
        break;

        case 'c':
            usingInitCaching = true;
            break;
        case 'u':
            UdoPackagePath = optarg;
            break;
        case 'x':
            cpuFixedPointMode = true;
            break;
        default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments" << std::endl;
                std::exit(FAILURE);
        }
    }

    // Check if given arguments represent valid files.
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList)
    {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        return EXIT_FAILURE;
    }

    // Check if given buffer type is valid 
    int bufferType;
    int bitWidth = 0;
    if (bufferTypeStr == "USERBUFFER_FLOAT")
    {
        bufferType = USERBUFFER_FLOAT;
    }
    else if (bufferTypeStr == "USERBUFFER_TF8")
    {
        bufferType = USERBUFFER_TF8;
        bitWidth = 8;
    }
    else if (bufferTypeStr == "USERBUFFER_TF16")
    {
        bufferType = USERBUFFER_TF16;
        bitWidth = 16;
    }
    else if (bufferTypeStr == "ITENSOR")
    {
        bufferType = ITENSOR;
    }
    else
    {
        std::cout << "Buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
        return EXIT_FAILURE;
    }

    // Check if given user buffer source type is valid.
    int userBufferSourceType;
    // CPUBUFFER / GLBUFFER supported only for USERBUFFER_FLOAT
    if (bufferType == USERBUFFER_FLOAT)
    {
        if (userBufferSourceStr == "CPUBUFFER")
        {
            userBufferSourceType = CPUBUFFER;
        }
        else if (userBufferSourceStr == "GLBUFFER")
        {
#ifndef ENABLE_GL_BUFFER
            std::cout << "GLBUFFER mode is only for supported on Android OS." << std::endl;
            return EXIT_FAILURE;
#endif
            userBufferSourceType = GLBUFFER;
        }
        else
        {
            std::cout
                  << "Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details"
                  << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (staticQuantizationStr == "true")
    {
        staticQuantization = true;
    } else if (staticQuantizationStr == "false")
    {
        staticQuantization = false;
    }
    else
    {
        std::cout << "Static quantization value is not valid. Please run snpe-sample with the -h flag for more details"
             << std::endl;
        return EXIT_FAILURE;
    }

    // Check if both runtimelist and runtime are passed in.
    if(runtimeSpecified && runtimeList.empty() == false)
    {
        std::cout << "Invalid option cannot mix runtime order -l with runtime -r " << std::endl;
        std::exit(FAILURE);
    }

    /*
    打开包含要执行的网络的 DL 容器 从现在打开的容器创建 SNPE 网络的实例。
    SNPE 提供的工厂函数允许指定网络的哪些层应作为输出返回，以及网络是否应在 CPU 或 GPU 上运行。
    运行时可用性 API 允许查询运行时支持。
    如果选定的运行时不可用，我们将发出警告并继续，期望在 SNPE 网络创建时捕获无效配置。
     */
    if(runtimeSpecified)
    {
        runtime = checkRuntime(runtime, staticQuantization);
    }

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);

    if (container == nullptr)
    {
        std::cerr << "Error while opening the container file." << std::endl;
        return EXIT_FAILURE;
    }

    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8 || bufferType == USERBUFFER_TF16);

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::PlatformConfig platformConfig;

    // load UDO package
    if (false == loadUDOPackage(UdoPackagePath))
    {
        std::cerr << "Failed to load UDO Package(s)." << std::endl;
        return EXIT_FAILURE;
    }

    snpe = setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, platformConfig, usingInitCaching, cpuFixedPointMode);

    if (snpe == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       return EXIT_FAILURE;
    }
    if (usingInitCaching)
    {
      if (container->save(dlc))
       {
          std::cout << "Saved container into archive successfully" << std::endl;
       }
       else
       {
          std::cout << "Failed to save container into archive" << std::endl;
       }
    }

    // Check the batch size for the container.
    // SNPE 1.16.0 and newer assumes the first dimension of the tensor shape is the batch size.

    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
    std::cout << "batchSize = " << batchSize << std::endl;

    auto t_shape = tensorShape.getDimensions();
    std::cout << *(t_shape + 1) << std::endl;
    std::cout << *(t_shape + 2) << std::endl;
    std::cout << *(t_shape + 3) << std::endl;

    std::vector<std::vector<std::string>> inputs = preprocessInput(inputFile, batchSize);
    return SUCCESS;
}
