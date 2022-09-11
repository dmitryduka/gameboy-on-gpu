# gameboy-on-gpu
Simple Gameboy DMG-01 emulator running entirely on GPU

![Alt text](images/160gameboys.png?raw=true "160 gameboy instances")

- no sound support
- no input support
  - it's easy to add, though keep in mind that controlling multiple instances of the the emulator, each running its own ROM is tricky 
- only supports MBC1 ROMs (i.e. no memory mappers)
- contains CPU/PPU bugs
  - as demonstrated by blargg's test ROMs
  - but it's enough to run simple ROMs (Tetris, Mario, test ROMs, etc)

This repository doesn't contain build scripts.
Dependencies are:

- CUDA
- SDL2
- spdlog

The central piece is the CUDA kernel "emulateGpu" which is called continuously in the main thread. Each call to this kernel advances the emulator state by 16ms forward. The state is persistent on the GPU between runs. There is not CUDA<->Graphics interop, because SDL2 doesn't support it. In principle, no emulator data should be sent to or from the GPU while the emulator is running, but this implementation copies the framebuffer from GPU to CPU and then back because of the API limitations. A different implementation (DX11/DX12/Vulkan, not SDL2) could leave the data on the GPU. "config.h" contains definitions for EMU_HEIGHT/EMU_WIDTH variables which define a grid of individual Gameboy emulators - currently it's configured to run a 8x4 grid, each running it's own ROM. This both configures the location in the framebuffer for each instance, as well as CUDA kernel scheduling parameters (e.g. 4 warps, 8 threads each). There is no good reason why these parameters affect both the rendering configuration and the kernel scheduling configuration at the same time. Obvisouly it's trivial to change this.

# Performance numbers

GPU: RTX 3070 Mobility
Time is per single kernel run, which emulates 16ms (Gameboy time)

1 instance - 25ms (1 warp, 1 thread)
1K instances - 60ms (1024 warps, 1 thread each, SM hardware warp schedulers *are* efficient)
1K instances - 85ms (32 warps, 32 threads each)
16K instances - 670ms (512 warps, 32 threads each)