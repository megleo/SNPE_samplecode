#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.h"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> &container,
                                                zdl::DlSystem::Runtime_t runtime,
                                                zdl::DlSystem::RuntimeList runtimeList,
                                                bool useUserSuppliedBuffers,
                                                zdl::DlSystem::PlatformConfig platformConfig,
                                                bool useCacheing,
                                                bool cpuFixedPointMode)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if (runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    snpe = snpeBuilder.setOutputLayers({})
        .setRuntimeProcessorOrder(runtimeList)
        .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
        .setPlatformConfig(platformConfig)
        .setInitCacheMode(useCacheing)
        .setCpuFixedPointMode(cpuFixedPointMode)
        .build();
    return snpe;
}