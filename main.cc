#include "config.h"
#if ENABLE_GPU == 1
#include <cuda_runtime.h>
#endif
#include <spdlog/spdlog.h>
#include <SDL.h>
#include <vector>
#include <algorithm>
#include <thread>

#include "cartridge.h"

using namespace std::chrono_literals;

void setup_logging()
{
    spdlog::set_pattern("[%c %z] [%^%L%$] %v");
    spdlog::set_level(spdlog::level::debug);
}

#if ENABLE_GPU == 1
void initGpuData(const uint8_t** roms, uint64_t roms_count);
cudaError_t emulateGpu(Gameboy* data, uint8_t* display);
#endif

int main(int, char**)
{
    setup_logging();
    bool quit = false;

#if ENABLE_GPU == 1
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
        spdlog::error("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
#endif

    SDL_Event event;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("gb", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_RendererInfo info{};
    SDL_GetRendererInfo(renderer, &info);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, FB_WIDTH * EMU_WIDTH, FB_HEIGHT * EMU_HEIGHT);

    {
        std::vector<Cartridge> roms;
        roms.emplace_back("roms\\tetris-world.gb");
        roms.emplace_back("roms\\dr-mario.gb");
        roms.emplace_back("roms\\yakuman.gb");
        roms.emplace_back("roms\\test-rom.gb");
        std::vector<const uint8_t*> roms_ptrs;
        roms_ptrs.resize(roms.size());
        std::transform(roms.begin(), roms.end(), roms_ptrs.begin(), [](const Cartridge& c) { return c.get_rom().data(); });
        initGpuData(roms_ptrs.data(), roms_ptrs.size());
    }
    std::vector<uint8_t> display;
    display.resize(FB_WIDTH * FB_HEIGHT * 3 * EMU_WIDTH * EMU_HEIGHT);
    while (!quit)
    {
#if ENABLE_GPU == 1
        const auto t1 = std::chrono::high_resolution_clock::now();
        emulateGpu(display.data());
        const auto t2 = std::chrono::high_resolution_clock::now();
        spdlog::info("CUDA: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
#endif
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        SDL_UpdateTexture(texture, nullptr, display.data(), EMU_WIDTH * FB_WIDTH * 3);

        while (SDL_PollEvent(&event))
        {
            SDL_PumpEvents();
            quit = event.type == SDL_QUIT;
        }

        // 60hz
        //std::this_thread::sleep_for(16ms);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

#if ENABLE_GPU == 1
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
#endif

    return 0;
}