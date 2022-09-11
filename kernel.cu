#include "config.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <array>
#include <stdio.h>

class Gameboy;
class GameboyCuda;

namespace
{
    __constant__ const uint8_t CYCLES_NORMAL[256] = {
        1, 3, 2, 2, 1, 1, 2, 1, 5, 2, 2, 2, 1, 1, 2, 1,
        1, 3, 2, 2, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1,
        2, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1,
        2, 3, 2, 2, 3, 3, 3, 1, 2, 2, 2, 2, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        2, 3, 3, 4, 3, 4, 2, 4, 2, 4, 3, 0, 3, 6, 2, 4,
        2, 3, 3, 0, 3, 4, 2, 4, 2, 4, 3, 0, 3, 0, 2, 4,
        3, 3, 2, 0, 0, 4, 2, 4, 4, 1, 4, 0, 0, 0, 2, 4,
        3, 3, 2, 1, 0, 4, 2, 4, 3, 2, 4, 1, 0, 0, 2, 4
    };

    __constant__ const uint8_t CYCLES_BRANCHED[256] = {
        1, 3, 2, 2, 1, 1, 2, 1, 5, 2, 2, 2, 1, 1, 2, 1,
        1, 3, 2, 2, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1,
        3, 3, 2, 2, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1,
        3, 3, 2, 2, 3, 3, 3, 1, 3, 2, 2, 2, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        5, 3, 4, 4, 6, 4, 2, 4, 5, 4, 4, 0, 6, 6, 2, 4,
        5, 3, 4, 0, 6, 4, 2, 4, 5, 4, 4, 0, 6, 0, 2, 4,
        3, 3, 2, 0, 0, 4, 2, 4, 4, 1, 4, 0, 0, 0, 2, 4,
        3, 3, 2, 1, 0, 4, 2, 4, 3, 2, 4, 1, 0, 0, 2, 4
    };

    __constant__ const uint8_t CYCLES_2B[256] = {
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2,
        2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2,
        2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2,
        2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2
    };

    __constant__ const uint8_t boot[256] = {
        0x31, 0xFE, 0xFF, 0xAF, 0x21, 0xFF, 0x9F, 0x32, 0xCB, 0x7C, 0x20, 0xFB, 0x21, 0x26, 0xFF, 0x0E,
        0x11, 0x3E, 0x80, 0x32, 0xE2, 0x0C, 0x3E, 0xF3, 0xE2, 0x32, 0x3E, 0x77, 0x77, 0x3E, 0xFC, 0xE0,
        0x47, 0x11, 0x04, 0x01, 0x21, 0x10, 0x80, 0x1A, 0xCD, 0x95, 0x00, 0xCD, 0x96, 0x00, 0x13, 0x7B,
        0xFE, 0x34, 0x20, 0xF3, 0x11, 0xD8, 0x00, 0x06, 0x08, 0x1A, 0x13, 0x22, 0x23, 0x05, 0x20, 0xF9,
        0x3E, 0x19, 0xEA, 0x10, 0x99, 0x21, 0x2F, 0x99, 0x0E, 0x0C, 0x3D, 0x28, 0x08, 0x32, 0x0D, 0x20,
        0xF9, 0x2E, 0x0F, 0x18, 0xF3, 0x67, 0x3E, 0x64, 0x57, 0xE0, 0x42, 0x3E, 0x91, 0xE0, 0x40, 0x04,
        0x1E, 0x02, 0x0E, 0x0C, 0xF0, 0x44, 0xFE, 0x90, 0x20, 0xFA, 0x0D, 0x20, 0xF7, 0x1D, 0x20, 0xF2,
        0x0E, 0x13, 0x24, 0x7C, 0x1E, 0x83, 0xFE, 0x62, 0x28, 0x06, 0x1E, 0xC1, 0xFE, 0x64, 0x20, 0x06,
        0x7B, 0xE2, 0x0C, 0x3E, 0x87, 0xE2, 0xF0, 0x42, 0x90, 0xE0, 0x42, 0x15, 0x20, 0xD2, 0x05, 0x20,
        0x4F, 0x16, 0x20, 0x18, 0xCB, 0x4F, 0x06, 0x04, 0xC5, 0xCB, 0x11, 0x17, 0xC1, 0xCB, 0x11, 0x17,
        0x05, 0x20, 0xF5, 0x22, 0x23, 0x22, 0x23, 0xC9, 0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B,
        0x03, 0x73, 0x00, 0x83, 0x00, 0x0C, 0x00, 0x0D, 0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E,
        0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99, 0xBB, 0xBB, 0x67, 0x63, 0x6E, 0x0E, 0xEC, 0xCC,
        0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E, 0x3C, 0x42, 0xB9, 0xA5, 0xB9, 0xA5, 0x42, 0x3C,
        0x21, 0x04, 0x01, 0x11, 0xA8, 0x00, 0x1A, 0x13, 0xBE, 0x00, 0x00, 0x23, 0x7D, 0xFE, 0x34, 0x20,
        0xF5, 0x06, 0x19, 0x78, 0x86, 0x23, 0x05, 0x20, 0xFB, 0x86, 0x00, 0x00, 0x3E, 0x01, 0xE0, 0x50
    };

    enum class VideoMode {
        eAccessOAM,
        eAccessVRAM,
        eHBlank,
        eVBlank,
    };

    constexpr uint64_t CLOCKS_PER_HBLANK = 204;
    constexpr uint64_t CLOCKS_PER_SCANLINE_OAM = 80;
    constexpr uint64_t CLOCKS_PER_SCANLINE_VRAM = 172;
    constexpr uint64_t CLOCKS_PER_SCANLINE = (CLOCKS_PER_SCANLINE_OAM + CLOCKS_PER_SCANLINE_VRAM + CLOCKS_PER_HBLANK);
    constexpr uint64_t CLOCKS_PER_VBLANK = 4560;
    constexpr uint64_t SCANLINES_PER_FRAME = 144;
    constexpr uint16_t TILE_BYTES = 2 * 8;
    constexpr uint16_t BG_MAP_SIZE = 256;
    constexpr uint16_t SPRITE_BYTES = 4;
    constexpr uint32_t TILES_PER_LINE = 32;
    constexpr uint32_t TILE_HEIGHT_PX = 8;
    constexpr uint32_t TILE_WIDTH_PX = 8;
    constexpr uint16_t TILE_SET_ZERO_ADDRESS = 0x8000;
    constexpr uint16_t TILE_SET_ONE_ADDRESS = 0x8800;
    constexpr uint16_t TILE_MAP_ZERO_ADDRESS = 0x9800;
    constexpr uint16_t TILE_MAP_ONE_ADDRESS = 0x9C00;

    __device__ inline bool in_range(uint16_t addr, uint16_t min, uint16_t max)
    {
        return addr >= min && addr <= max;
    }

    enum class Flags : uint8_t
    {
        eCarry = 4,
        eHalfCarry = 5,
        eSubtract = 6,
        eZero = 7
    };

    __device__ inline void set_bit(uint8_t& value, uint8_t bit, bool set)
    {
        if (set)
            value |= 1 << bit;
        else
            value &= ~(1 << bit);
    }

    __device__ inline bool is_set(uint8_t& value, uint8_t bit)
    {
        return value & (1 << bit);
    }

    __device__ inline void set_bit(uint8_t& value, Flags bit, bool set)
    {
        set_bit(value, static_cast<uint8_t>(bit), set);
    }

    __device__ inline bool is_set(uint8_t& value, Flags bit)
    {
        return is_set(value, static_cast<uint8_t>(bit));
    }

    enum class Color {
        eWhite,
        eLightGray,
        eDarkGray,
        eBlack,
    };

    __device__ uint8_t get_gray(Color c)
    {
        if (c == Color::eWhite)
            return 0x9B;
        else if (c == Color::eLightGray)
            return 0x8B;
        else if (c == Color::eDarkGray)
            return 0x48;
        else if (c == Color::eBlack)
            return 0x0F;
        return 0x00;
    }

    __device__ Color get_color(uint8_t value)
    {
        switch (value) {
        case 0:
            return Color::eWhite;
        case 1:
            return Color::eLightGray;
        case 2:
            return Color::eDarkGray;
        case 3:
            return Color::eBlack;
        }
        return Color::eBlack;
    }

    const uint16_t INT_VBLANK = 0x40;
    const uint16_t INT_LCDC_STATUS = 0x48;
    const uint16_t INT_TIMER = 0x50;
    const uint16_t INT_SERIAL = 0x58;
    const uint16_t INT_JOY = 0x60;
}

enum class RAMSize {
    eNone,
    eKB2,
    eKB8,
    eKB32,
    eKB128,
    eKB64,
};

enum class ROMSize {
    eKB32,
    eKB64,
    eKB128,
    eKB256,
    eKB512,
    eMB1,
    eMB2,
    eMB4,
    eMB1p1,
    eMB1p2,
    eMB1p5,
};

enum class CartridgeType {
    eROMOnly,
    eMBC1,
    eMBC2,
    eMBC3,
    eMBC4,
    eMBC5
};

class Cartridge {
public:
    __device__ uint8_t read(const uint16_t& address) const
    {
        return rom[address];
    }
    uint8_t rom[32 * 1024];
    uint8_t info[112];
};

struct CPUState
{
    union Register
    {
        uint16_t v;
        struct { uint8_t f, a; };
        struct { uint8_t c, b; };
        struct { uint8_t e, d; };
        struct { uint8_t l, h; };
    };

    union Registers
    {
        struct
        {
            Register bc, de, hl, af;
        };
        uint8_t r[8];
    } rf;
    bool interrupts_enabled{}, branched{}, halt{};
    uint16_t pc{}, sp{};
    uint8_t interrupt_flag{}, interrupt_enabled{};
};

class PPUCuda
{
public:
    __device__ void tick(uint64_t cycles);

    __device__ uint8_t read(uint16_t address)
    {
        return vram[address];
    }
    __device__ void write(uint16_t address, uint8_t value)
    {
        vram[address] = value;
    }

    uint8_t control_byte{}, line{}, ly_cmp{}, lcd_status{}, scroll_x{},
        scroll_y{}, window_x{}, window_y{}, bg_palette{}, sprite_palette_0{}, sprite_palette_1{};

    __device__ void set_pixel(uint16_t x, uint16_t y, uint8_t gray)
    {
        const auto global_x = x + FB_WIDTH * threadIdx.x;
        const auto global_y = y + FB_HEIGHT * blockIdx.x;
        const auto base = (global_x + global_y * FB_WIDTH * EMU_WIDTH) * 3;
        display[base] = gray;
        display[base + 1] = gray + 41;
        display[base + 2] = gray;
    }

    __device__ uint8_t get_pixel(uint16_t x, uint16_t y)
    {
        const auto global_x = x + FB_WIDTH * threadIdx.x;
        const auto global_y = y + FB_HEIGHT * blockIdx.x;
        const auto base = (global_x + global_y * FB_WIDTH * EMU_WIDTH) * 3;
        return display[base];
    }


    GameboyCuda* gb;
    uint64_t cycle_counter{};
    VideoMode mode{ VideoMode::eAccessOAM };
    uint8_t* display;
    uint8_t vram[16 * 1024];
};

class MMUCuda
{
public:
    __device__ uint8_t read(uint16_t a);
    __device__ void write(uint16_t a, uint8_t value);

    bool startup{ true };

    GameboyCuda* gb;
    uint8_t work_ram[0x2000];
    uint8_t oam_ram[160];
    uint8_t high_ram[128];
};

class CPUCuda
{
public:
    __device__ uint8_t tick();
    __device__ void stack_push(uint16_t value);
    __device__ uint16_t stack_pop();
    __device__ uint16_t read_word_at_pc();
    __device__ uint8_t execute_instruction(uint8_t inst);
    __device__ void set_interrupt_flag(uint8_t no, bool value);
    __device__ void set_interrupt_enabled(uint8_t value);
    __device__ uint8_t get_interrupt_enabled() const;
    __device__ uint8_t get_interrupt_flag() const;

    GameboyCuda* gb;
    CPUState state;
    uint64_t cycles_emulated{};
};

class GameboyCuda
{
public:
    __device__ void tick(uint16_t ms);

    friend class PPU;
    friend class MMUCuda;
    friend class CPUCuda;
    friend class Input;

    Cartridge cartridge;
    CPUCuda cpu;
    PPUCuda ppu;
    MMUCuda mmu;
    //Input joypad;
};

__device__ uint8_t MMUCuda::read(uint16_t a)
{
    // Memory map:
    // 0-256b                            (256b)                         Boot ROM (only during startup)
    // 0-16k                             (16k)                          Game ROM Bank 0
    // 16k-32k                           (16k)                          Game ROM Bank N (mapper select the specific bank)
    // 32k-38911b                        (6143b)                        Tile RAM
    // 38912b-40959b                     (2k)                           Background Map
    // 40960b-49151b                     (8k)                           Cartridge RAM (optionally)
    // 49152b-57343b                     (4k)                           Working RAM (external RAM)
    // 57344b-65023b                     (7679b)                        Echo RAM (unused)
    // 65025b-65183b                     (160b)                         OAM
    // 65184b-65279b                     (95b)                          Unused
    // 65280b-65407b                     (127b)                         I/O registers
    // 65408b-65534b                     (126b)                         High RAM (on-chip RAM)
    // 65535b                            (1b)                           Interrupt enabled register

    // cartidge or boot
    if (in_range(a, 0, 0x7FFF))
    {
        if (a <= 0xFF && startup)
            return boot[a];
        // mapper magic to read banks other than 0 happens inside the cartridge
        return gb->cartridge.read(a);
    }
    // Tile RAM & Background Map
    else if (in_range(a, 0x8000, 0x9FFF))
        return gb->ppu.read(a - 0x8000);
    // Cartridge RAM
    else if (in_range(a, 0xA000, 0xBFFF))
    {
        // TODO
    }
    else if (in_range(a, 0xC000, 0xDFFF))
        return work_ram[a - 0xC000];
    else if (in_range(a, 0xE000, 0xFDFF))
        return work_ram[a - 0xE000];
    else if (in_range(a, 0xFE00, 0xFE9F))
        return oam_ram[a - 0xFE00];
    else if (in_range(a, 0xFF00, 0xFF7F))
    {
        switch (a)
        {
        case 0xFF00:
            return 0xFF;
            //return gb->joypad.get_input_register();
        case 0xFF0F:
            return gb->cpu.get_interrupt_flag();
        case 0xFF40:
            return gb->ppu.control_byte;
        case 0xFF41:
            return gb->ppu.lcd_status;
        case 0xFF42:
            return gb->ppu.scroll_y;
        case 0xFF43:
            return gb->ppu.scroll_x;
        case 0xFF44:
            return gb->ppu.line;
        case 0xFF45:
            return gb->ppu.ly_cmp;
        case 0xFF47:
            return gb->ppu.bg_palette;
        case 0xFF48:
            return gb->ppu.sprite_palette_0;
        case 0xFF49:
            return gb->ppu.sprite_palette_1;
        case 0xFF4A:
            return gb->ppu.window_y;
        case 0xFF4B:
            return gb->ppu.window_x;
        case 0xFF50:
            return startup ? 1 : 0;
        }
    }
    else if (in_range(a, 0xFF80, 0xFFFE))
        return high_ram[a - 0xFF80];
    else if (a == 0xFFFF)
        return gb->cpu.get_interrupt_enabled();

    return 0xFF;
}

__device__ void MMUCuda::write(uint16_t a, uint8_t value)
{
    if (in_range(a, 0xFF80, 0xFFFE))
        high_ram[a - 0xFF80] = value;
    else if (in_range(a, 0x8000, 0x9FFF))
        gb->ppu.write(a - 0x8000, value);
    else if (in_range(a, 0xC000, 0xDFFF))
        work_ram[a - 0xC000] = value;
    else if (in_range(a, 0xFE00, 0xFE9F))
        oam_ram[a - 0xFE00] = value;
    else if (in_range(a, 0xFF00, 0xFF7F))
    {
        switch (a)
        {
        case 0xFF00:
            //gb->joypad.write(value);
            break;
        case 0xFF40:
            gb->ppu.control_byte = value;
            break;
        case 0xFF41:
            gb->ppu.lcd_status = value;
            break;
        case 0xFF42:
            gb->ppu.scroll_y = value;
            break;
        case 0xFF43:
            gb->ppu.scroll_x = value;
            break;
        case 0xFF44:
            gb->ppu.line = 0;
            break;
        case 0xFF45:
            gb->ppu.ly_cmp = value;
            break;
        case 0xFF46:
        {
            const uint16_t start_address = value * 0x100;
            for (uint8_t i = 0; i < 160; i++)
                write(0xFE00 + i, read(start_address + i));
        }
        break;
        case 0xFF47:
            gb->ppu.bg_palette = value;
            break;
        case 0xFF48:
            gb->ppu.sprite_palette_0 = value;
            break;
        case 0xFF49:
            gb->ppu.sprite_palette_1 = value;
            break;
        case 0xFF4A:
            gb->ppu.window_y = value;
            break;
        case 0xFF4B:
            gb->ppu.window_x = value;
            break;
        case 0xFF50:
            startup = value ? false : true;
            break;
        }
    }
    else if (in_range(a, 0xE000, 0xFDFF))
        work_ram[a - 0xE000] = value;
    else if (a == 0xFFFF)
        gb->cpu.set_interrupt_enabled(value);
}


__device__ uint8_t CPUCuda::tick()
{
    // handle interrupts first
    if (state.interrupts_enabled)
    {
        const auto active_interrupts = state.interrupt_flag & state.interrupt_enabled;
        if (active_interrupts)
        {
            state.halt = false;
            stack_push(state.pc);

            uint16_t int_index{};
            for (const auto inter : { INT_VBLANK, INT_LCDC_STATUS, INT_TIMER, INT_SERIAL, INT_JOY })
            {
                if (active_interrupts & (1 << int_index))
                {
                    state.interrupt_flag &= ~(1 << int_index);
                    state.pc = inter;
                    state.interrupts_enabled = false;
                    break;
                }
                int_index += 1;
            }
        }
    }
    // if halted, exit
    if (state.halt)
        return 1;
    // otherwise, fetch next byte and execute it
    return execute_instruction(gb->mmu.read(state.pc++));
}

__device__ void CPUCuda::stack_push(uint16_t value)
{
    gb->mmu.write(--state.sp, value >> 8);
    gb->mmu.write(--state.sp, value & 0xFF);
}

__device__ uint16_t CPUCuda::stack_pop()
{
    const auto l = gb->mmu.read(state.sp++);
    const auto h = gb->mmu.read(state.sp++);
    return (uint16_t(h) << 8) + l;
}

__device__ uint16_t CPUCuda::read_word_at_pc()
{
    uint16_t word = gb->mmu.read(state.pc++);
    word += uint16_t(gb->mmu.read(state.pc++)) << 8;
    return word;
}

__device__ uint8_t CPUCuda::execute_instruction(uint8_t inst)
{
    // NOTE: this is handy https://www.pastraiser.com/cpu/gameboy/gameboy_opcodes.html
    state.branched = false;
    uint8_t next_inst{};

    const auto set_flags = [&](auto reg)
    {
        set_bit(state.rf.af.f, Flags::eZero, reg == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, (reg & 0x0F) == 0);
    };

    const auto clear_flags = [&]()
    {
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, false);
    };

    const auto add_hl = [&](auto reg)
    {
        const auto prev = state.rf.hl.v;
        const int result = state.rf.hl.v + reg;
        state.rf.hl.v = static_cast<uint16_t>(result);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, (prev & 0xFFF) + (reg & 0xFFF) > 0xFFF);
        set_bit(state.rf.af.f, Flags::eCarry, result & 0x10000);
    };

    const auto add_a = [&](auto val)
    {
        const auto old = state.rf.af.a;
        const int result = state.rf.af.a + val;
        state.rf.af.a = static_cast<uint8_t>(result);
        set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, ((old & 0xF) + (val & 0xF)) > 0xF);
        set_bit(state.rf.af.f, Flags::eCarry, result & 0x100);
    };

    const auto adc_a = [&](auto val)
    {
        const auto old = state.rf.af.a;
        const bool carry = is_set(state.rf.af.f, Flags::eCarry);
        const uint16_t result = old + val + (carry ? 1 : 0);
        state.rf.af.a = static_cast<uint8_t>(result);

        set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, ((old & 0xF) + (val & 0xF) + (carry ? 1 : 0)) > 0xF);
        set_bit(state.rf.af.f, Flags::eCarry, result > 0xFF);
    };

    const auto sub_a = [&](auto val)
    {
        const auto old = state.rf.af.a;
        state.rf.af.a += val;
        set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, true);
        set_bit(state.rf.af.f, Flags::eHalfCarry, (int8_t(old & 0xF) - int8_t(val & 0xF)) < 0);
        set_bit(state.rf.af.f, Flags::eCarry, old < val);
    };

    const auto sbc_a = [&](auto val)
    {
        const auto old = state.rf.af.a;
        const bool carry = is_set(state.rf.af.f, Flags::eCarry);
        const int result = old - val - (carry ? 1 : 0);
        state.rf.af.a = static_cast<uint8_t>(result);

        set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, true);
        set_bit(state.rf.af.f, Flags::eHalfCarry, (int8_t(old & 0xF) - int8_t(val & 0xF) - (carry ? 1 : 0)) < 0);
        set_bit(state.rf.af.f, Flags::eCarry, result < 0);
    };

    const auto and_a = [&](auto val)
    {
        state.rf.af.a &= val;
        set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, true);
        set_bit(state.rf.af.f, Flags::eCarry, false);
    };

    const auto xor_a = [&](auto val)
    {
        state.rf.af.a ^= val;
        set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
        clear_flags();
    };

    const auto or_a = [&](auto val)
    {
        state.rf.af.a |= val;
        set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
        clear_flags();
    };

    const auto cp = [&](auto val)
    {
        uint8_t result = static_cast<uint8_t>(state.rf.af.a - val);
        set_bit(state.rf.af.f, Flags::eZero, result == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, true);
        set_bit(state.rf.af.f, Flags::eHalfCarry, ((state.rf.af.a & 0xf) - (val & 0xf)) < 0);
        set_bit(state.rf.af.f, Flags::eCarry, state.rf.af.a < val);
    };

    const auto bit = [&](auto reg, uint8_t bit)
    {
        set_bit(state.rf.af.f, Flags::eZero, !is_set(reg, bit));
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, true);
    };

    const auto swap = [&](auto reg) -> uint8_t
    {
        const uint8_t res = (((reg & 0x0F) << 4) | (reg & 0xF0) >> 4);
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, false);
        return res;
    };

    const auto rlc = [&](auto reg) -> uint8_t
    {
        const bool carry = is_set(reg, 7);
        uint8_t res = static_cast<uint8_t>((reg << 1) | (carry ? 1 : 0));
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, carry);
        return res;
    };

    const auto rrc = [&](auto reg) -> uint8_t
    {
        const auto carry = is_set(reg, 0);
        uint8_t res = static_cast<uint8_t>((reg >> 1) | (carry ? 1 << 7 : 0));
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, carry);
        return res;
    };

    const auto rl = [&](auto reg) -> uint8_t
    {
        const auto carry = is_set(state.rf.af.f, 0);
        const auto will_carry = is_set(reg, 7);
        const uint8_t res = static_cast<uint8_t>(reg << 1 | (carry ? 1 : 0));
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, will_carry);
        return res;
    };

    const auto rr = [&](auto reg) -> uint8_t
    {
        const auto carry = is_set(state.rf.af.f, 0);
        const auto will_carry = is_set(reg, 7);
        const uint8_t res = static_cast<uint8_t>(reg >> 1 | (carry ? 1 << 7 : 0));
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, will_carry);
        return res;
    };

    const auto sla = [&](auto reg) -> uint8_t
    {
        const auto carry = is_set(reg, 0);
        const uint8_t res = static_cast<uint8_t>(reg << 1);
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, carry);
        return res;
    };

    const auto sra = [&](auto reg) -> uint8_t
    {
        const auto carry = is_set(reg, 0);
        const auto top_bit = is_set(reg, 7);
        const uint8_t res = static_cast<uint8_t>(reg >> 1 | (top_bit ? 1 << 7 : 0));
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, carry);
        return res;
    };

    const auto srl = [&](auto reg) -> uint8_t
    {
        const auto low_bit = is_set(reg, 0);
        const uint8_t res = static_cast<uint8_t>(reg >> 1);
        set_bit(state.rf.af.f, Flags::eZero, res == 0);
        set_bit(state.rf.af.f, Flags::eSubtract, false);
        set_bit(state.rf.af.f, Flags::eHalfCarry, false);
        set_bit(state.rf.af.f, Flags::eCarry, low_bit);
        return res;
    };

    if (inst != 0xCB)
    {
        switch (inst)
        {
        case 0x01:
            state.rf.bc.v = read_word_at_pc();
            break;
        case 0x02:
            gb->mmu.write(state.rf.bc.v, state.rf.af.a);
            break;
        case 0x03:
            state.rf.bc.v++;
            break;
        case 0x04:
            state.rf.bc.b++;
            set_flags(state.rf.bc.b);
            break;
        case 0x05:
            state.rf.bc.b--;
            set_flags(state.rf.bc.b);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x06:
            state.rf.bc.b = gb->mmu.read(state.pc++);
            break;
        case 0x07:
            {
                state.rf.af.a = rlc(state.rf.af.a);
                set_bit(state.rf.af.f, Flags::eZero, false);
            }
            break;
        case 0x08:
            {
                const auto word = read_word_at_pc();
                gb->mmu.write(word, state.sp & 0xFF);
                gb->mmu.write(word + 1, state.sp >> 8);
            }
            break;
        case 0x09:
            add_hl(state.rf.bc.v);
            break;
        case 0x0A:
            state.rf.af.a = gb->mmu.read(state.rf.bc.v);
            break;
        case 0x0B:
            state.rf.bc.v--;
            break;
        case 0x0C:
            state.rf.bc.c++;
            set_flags(state.rf.bc.c);
            break;
        case 0x0D:
            state.rf.bc.c--;
            set_flags(state.rf.bc.c);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x0E:
            state.rf.bc.c = gb->mmu.read(state.pc++);
            break;
        case 0x0F:
            state.rf.af.a = rrc(state.rf.af.a);
            set_bit(state.rf.af.f, Flags::eZero, false);
            break;
        case 0x10:
            break;
        case 0x11:
            state.rf.de.v = read_word_at_pc();
            break;
        case 0x12:
            gb->mmu.write(state.rf.de.v, state.rf.af.a);
            break;
        case 0x13:
            state.rf.de.v++;
            break;
        case 0x14:
            state.rf.de.d++;
            set_flags(state.rf.de.d);
            break;
        case 0x15:
            state.rf.de.d--;
            set_flags(state.rf.de.d);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x16:
            state.rf.de.d = gb->mmu.read(state.pc++);
            break;
        case 0x17:
            {
                const auto carry = is_set(state.rf.af.f, Flags::eCarry);
                if (is_set(state.rf.af.a, 7))
                    set_bit(state.rf.af.f, Flags::eCarry, true);
                state.rf.af.a = (state.rf.af.a << 1) | (carry ? 1 : 0);
                set_bit(state.rf.af.f, Flags::eZero, false);
                set_bit(state.rf.af.f, Flags::eSubtract, false);
                set_bit(state.rf.af.f, Flags::eHalfCarry, false);
            }
            break;
        case 0x18:
            state.pc += static_cast<int8_t>(gb->mmu.read(state.pc++));
            break;
        case 0x19:
            add_hl(state.rf.de.v);
            break;
        case 0x1A:
            state.rf.af.a = gb->mmu.read(state.rf.de.v);
            break;
        case 0x1B:
            state.rf.de.v--;
            break;
        case 0x1C:
            state.rf.de.e++;
            set_flags(state.rf.de.e);
            break;
        case 0x1D:
            state.rf.de.e--;
            set_flags(state.rf.de.e);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x1E:
            state.rf.de.e = gb->mmu.read(state.pc++);
            break;
        case 0x1F:
            {
                const auto carry = is_set(state.rf.af.f, Flags::eCarry);
                if (is_set(state.rf.af.a, 0))
                    set_bit(state.rf.af.f, Flags::eCarry, true);
                state.rf.af.a = (state.rf.af.a >> 1) | (carry ? 1 << 7 : 0);
                set_bit(state.rf.af.f, Flags::eZero, false);
                set_bit(state.rf.af.f, Flags::eSubtract, false);
                set_bit(state.rf.af.f, Flags::eHalfCarry, false);
            }
            break;
        case 0x20:
            {
                const int8_t jmp = static_cast<int8_t>(gb->mmu.read(state.pc++));
                if (!is_set(state.rf.af.f, Flags::eZero))
                {
                    state.pc += jmp;
                    state.branched = true;
                }
            }
            break;
        case 0x21:
            state.rf.hl.v = read_word_at_pc();
            break;
        case 0x22:
            gb->mmu.write(state.rf.hl.v++, state.rf.af.a);
            break;
        case 0x23:
            state.rf.hl.v++;
            break;
        case 0x24:
            state.rf.hl.h++;
            set_flags(state.rf.hl.h);
            break;
        case 0x25:
            state.rf.hl.h--;
            set_flags(state.rf.hl.h);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x26:
            state.rf.hl.h = gb->mmu.read(state.pc++);
            break;
        case 0x27:
            {
                const auto subtract = is_set(state.rf.af.f, Flags::eSubtract);

                if (!subtract)
                {
                    if (is_set(state.rf.af.f, Flags::eCarry) || (state.rf.af.a > 0x99))
                    {
                        state.rf.af.a += 0x60;
                        set_bit(state.rf.af.f, Flags::eCarry, true);
                    }
                    if (is_set(state.rf.af.f, Flags::eHalfCarry) || ((state.rf.af.a & 0x0F) > 9))
                        state.rf.af.a += 0x06;
                }
                else
                {
                    if (is_set(state.rf.af.f, Flags::eCarry))
                    {
                        state.rf.af.a -= 0x60;
                        set_bit(state.rf.af.f, Flags::eCarry, true);
                    }
                    if (is_set(state.rf.af.f, Flags::eHalfCarry))
                        state.rf.af.a -= 0x06;
                }

                set_bit(state.rf.af.f, Flags::eHalfCarry, false);
                set_bit(state.rf.af.f, Flags::eZero, state.rf.af.a == 0);
            }
            break;
        case 0x28:
            {
                const int8_t jmp = static_cast<int8_t>(gb->mmu.read(state.pc++));
                if (is_set(state.rf.af.f, Flags::eZero))
                {
                    state.pc += jmp;
                    state.branched = true;
                }
            }
            break;
        case 0x29:
            add_hl(state.rf.hl.v);
            break;
        case 0x2A:
            state.rf.af.a = gb->mmu.read(state.rf.hl.v++);
            break;
        case 0x2B:
            state.rf.hl.v--;
            break;
        case 0x2C:
            state.rf.hl.l++;
            set_flags(state.rf.hl.l);
            break;
        case 0x2D:
            state.rf.hl.l--;
            set_flags(state.rf.hl.l);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x2E:
            state.rf.hl.l = gb->mmu.read(state.pc++);
            break;
        case 0x2F:
            state.rf.af.a = ~state.rf.af.a;
            set_bit(state.rf.af.f, Flags::eHalfCarry, true);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x30:
            {
                const int8_t jmp = static_cast<int8_t>(gb->mmu.read(state.pc++));
                if (!is_set(state.rf.af.f, Flags::eCarry))
                {
                    state.pc += jmp;
                    state.branched = true;
                }
            }
            break;
        case 0x31:
            state.sp = read_word_at_pc();
            break;
        case 0x32:
            gb->mmu.write(state.rf.hl.v--, state.rf.af.a);
            break;
        case 0x33:
            state.sp++;
            break;
        case 0x34:
            {
                uint8_t result = gb->mmu.read(state.rf.hl.v) + 1;
                gb->mmu.write(state.rf.hl.v, result);
                set_flags(result);
            }
            break;
        case 0x35:
            {
                uint8_t result = gb->mmu.read(state.rf.hl.v) - 1;
                gb->mmu.write(state.rf.hl.v, result);
                set_flags(result);
            }
            break;
        case 0x36:
            gb->mmu.write(state.rf.hl.v, gb->mmu.read(state.pc++));
            break;
        case 0x37:
            set_bit(state.rf.af.f, Flags::eCarry, true);
            set_bit(state.rf.af.f, Flags::eHalfCarry, false);
            set_bit(state.rf.af.f, Flags::eSubtract, false);
            break;
        case 0x38:
            {
                const int8_t jmp = static_cast<int8_t>(gb->mmu.read(state.pc++));
                if (is_set(state.rf.af.f, Flags::eCarry))
                {
                    state.pc += jmp;
                    state.branched = true;
                }
            }
            break;
        case 0x39:
            add_hl(state.sp);
            break;
        case 0x3A:
            state.rf.af.a = gb->mmu.read(state.rf.hl.v--);
            break;
        case 0x3B:
            state.sp--;
            break;
        case 0x3C:
            state.rf.af.a++;
            set_flags(state.rf.af.a);
            break;
        case 0x3D:
            state.rf.af.a--;
            set_flags(state.rf.af.a);
            set_bit(state.rf.af.f, Flags::eSubtract, true);
            break;
        case 0x3E:
            state.rf.af.a = gb->mmu.read(state.pc++);
            break;
        case 0x3F:
            set_bit(state.rf.af.f, Flags::eCarry, !is_set(state.rf.af.f, Flags::eCarry));
            set_bit(state.rf.af.f, Flags::eHalfCarry, false);
            set_bit(state.rf.af.f, Flags::eSubtract, false);
            break;
        case 0x41:
            state.rf.bc.b = state.rf.bc.c;
            break;
        case 0x42:
            state.rf.bc.b = state.rf.de.d;
            break;
        case 0x43:
            state.rf.bc.b = state.rf.de.e;
            break;
        case 0x44:
            state.rf.bc.b = state.rf.hl.h;
            break;
        case 0x45:
            state.rf.bc.b = state.rf.hl.l;
            break;
        case 0x46:
            state.rf.bc.b = gb->mmu.read(state.rf.hl.v);
            break;
        case 0x47:
            state.rf.bc.b = state.rf.af.a;
            break;
        case 0x48:
            state.rf.bc.c = state.rf.bc.b;
            break;
        case 0x4A:
            state.rf.bc.c = state.rf.de.d;
            break;
        case 0x4B:
            state.rf.bc.c = state.rf.de.e;
            break;
        case 0x4C:
            state.rf.bc.c = state.rf.hl.h;
            break;
        case 0x4D:
            state.rf.bc.c = state.rf.hl.l;
            break;
        case 0x4E:
            state.rf.bc.c = gb->mmu.read(state.rf.hl.v);
            break;
        case 0x4F:
            state.rf.bc.c = state.rf.af.a;
            break;
        case 0x50:
            state.rf.de.d = state.rf.bc.b;
            break;
        case 0x51:
            state.rf.de.d = state.rf.bc.c;
            break;
        case 0x53:
            state.rf.de.d = state.rf.de.e;
            break;
        case 0x54:
            state.rf.de.d = state.rf.hl.h;
            break;
        case 0x55:
            state.rf.de.d = state.rf.hl.l;
            break;
        case 0x56:
            state.rf.de.d = gb->mmu.read(state.rf.hl.v);
            break;
        case 0x57:
            state.rf.de.d = state.rf.af.a;
            break;
        case 0x58:
            state.rf.de.e = state.rf.bc.b;
            break;
        case 0x59:
            state.rf.de.e = state.rf.bc.c;
            break;
        case 0x5A:
            state.rf.de.e = state.rf.de.d;
            break;
        case 0x5C:
            state.rf.de.e = state.rf.hl.h;
            break;
        case 0x5D:
            state.rf.de.e = state.rf.hl.l;
            break;
        case 0x5E:
            state.rf.de.e = gb->mmu.read(state.rf.hl.v);
            break;
        case 0x5F:
            state.rf.de.e = state.rf.af.a;
            break;
        case 0x60:
            state.rf.hl.h = state.rf.bc.b;
            break;
        case 0x61:
            state.rf.hl.h = state.rf.bc.c;
            break;
        case 0x62:
            state.rf.hl.h = state.rf.de.d;
            break;
        case 0x63:
            state.rf.hl.h = state.rf.de.e;
            break;
        case 0x65:
            state.rf.hl.h = state.rf.hl.l;
            break;
        case 0x66:
            state.rf.hl.h = gb->mmu.read(state.rf.hl.v);
            break;
        case 0x67:
            state.rf.hl.h = state.rf.af.a;
            break;
        case 0x68:
            state.rf.hl.l = state.rf.bc.b;
            break;
        case 0x69:
            state.rf.hl.l = state.rf.bc.c;
            break;
        case 0x6A:
            state.rf.hl.l = state.rf.de.d;
            break;
        case 0x6B:
            state.rf.hl.l = state.rf.de.e;
            break;
        case 0x6C:
            state.rf.hl.l = state.rf.hl.h;
            break;
        case 0x6E:
            state.rf.hl.l = gb->mmu.read(state.rf.hl.v);
            break;
        case 0x6F:
            state.rf.hl.l = state.rf.af.a;
            break;
        case 0x70:
            gb->mmu.write(state.rf.hl.v, state.rf.bc.b);
            break;
        case 0x71:
            gb->mmu.write(state.rf.hl.v, state.rf.bc.c);
            break;
        case 0x72:
            gb->mmu.write(state.rf.hl.v, state.rf.de.d);
            break;
        case 0x73:
            gb->mmu.write(state.rf.hl.v, state.rf.de.e);
            break;
        case 0x74:
            gb->mmu.write(state.rf.hl.v, state.rf.hl.h);
            break;
        case 0x75:
            gb->mmu.write(state.rf.hl.v, state.rf.hl.l);
            break;
        case 0x76:
            state.halt = true;
            break;
        case 0x77:
            gb->mmu.write(state.rf.hl.v, state.rf.af.a);
            break;
        case 0x78:
            state.rf.af.a = state.rf.bc.b;
            break;
        case 0x79:
            state.rf.af.a = state.rf.bc.c;
            break;
        case 0x7A:
            state.rf.af.a = state.rf.de.d;
            break;
        case 0x7B:
            state.rf.af.a = state.rf.de.e;
            break;
        case 0x7C:
            state.rf.af.a = state.rf.hl.h;
            break;
        case 0x7D:
            state.rf.af.a = state.rf.hl.l;
            break;
        case 0x7E:
            state.rf.af.a = gb->mmu.read(state.rf.hl.v);
            break;
        case 0x80:
            add_a(state.rf.bc.b);
            break;
        case 0x81:
            add_a(state.rf.bc.c);
            break;
        case 0x82:
            add_a(state.rf.de.d);
            break;
        case 0x83:
            add_a(state.rf.de.e);
            break;
        case 0x84:
            add_a(state.rf.hl.h);
            break;
        case 0x85:
            add_a(state.rf.hl.l);
            break;
        case 0x86:
            add_a(gb->mmu.read(state.rf.hl.v));
            break;
        case 0x87:
            add_a(state.rf.af.a);
            break;
        case 0x88:
            adc_a(state.rf.bc.b);
            break;
        case 0x89:
            adc_a(state.rf.bc.c);
            break;
        case 0x8A:
            adc_a(state.rf.de.d);
            break;
        case 0x8B:
            adc_a(state.rf.de.e);
            break;
        case 0x8C:
            adc_a(state.rf.hl.h);
            break;
        case 0x8D:
            adc_a(state.rf.hl.l);
            break;
        case 0x8E:
            adc_a(gb->mmu.read(state.rf.hl.v));
            break;
        case 0x8F:
            adc_a(state.rf.af.a);
            break;
        case 0x90:
            sub_a(state.rf.bc.b);
            break;
        case 0x91:
            sub_a(state.rf.bc.c);
            break;
        case 0x92:
            sub_a(state.rf.de.d);
            break;
        case 0x93:
            sub_a(state.rf.de.e);
            break;
        case 0x94:
            sub_a(state.rf.hl.h);
            break;
        case 0x95:
            sub_a(state.rf.hl.l);
            break;
        case 0x96:
            sub_a(gb->mmu.read(state.rf.hl.v));
            break;
        case 0x97:
            sub_a(state.rf.af.a);
            break;
        case 0x98:
            sbc_a(state.rf.bc.b);
            break;
        case 0x99:
            sbc_a(state.rf.bc.c);
            break;
        case 0x9A:
            sbc_a(state.rf.de.d);
            break;
        case 0x9B:
            sbc_a(state.rf.de.e);
            break;
        case 0x9C:
            sbc_a(state.rf.hl.h);
            break;
        case 0x9D:
            sbc_a(state.rf.hl.l);
            break;
        case 0x9E:
            sbc_a(gb->mmu.read(state.rf.hl.v));
            break;
        case 0x9F:
            sbc_a(state.rf.af.a);
            break;
        case 0xA0:
            and_a(state.rf.bc.b);
            break;
        case 0xA1:
            and_a(state.rf.bc.c);
            break;
        case 0xA2:
            and_a(state.rf.de.d);
            break;
        case 0xA3:
            and_a(state.rf.de.e);
            break;
        case 0xA4:
            and_a(state.rf.hl.h);
            break;
        case 0xA5:
            and_a(state.rf.hl.l);
            break;
        case 0xA6:
            and_a(gb->mmu.read(state.rf.hl.v));
            break;
        case 0xA7:
            and_a(state.rf.af.a);
            break;
        case 0xA8:
            xor_a(state.rf.bc.b);
            break;
        case 0xA9:
            xor_a(state.rf.bc.c);
            break;
        case 0xAA:
            xor_a(state.rf.de.d);
            break;
        case 0xAB:
            xor_a(state.rf.de.e);
            break;
        case 0xAC:
            xor_a(state.rf.hl.h);
            break;
        case 0xAD:
            xor_a(state.rf.hl.l);
            break;
        case 0xAE:
            xor_a(gb->mmu.read(state.rf.hl.v));
            break;
        case 0xAF:
            xor_a(state.rf.af.a);
            break;
        case 0xB0:
            or_a(state.rf.bc.b);
            break;
        case 0xB1:
            or_a(state.rf.bc.c);
            break;
        case 0xB2:
            or_a(state.rf.de.d);
            break;
        case 0xB3:
            or_a(state.rf.de.e);
            break;
        case 0xB4:
            or_a(state.rf.hl.h);
            break;
        case 0xB5:
            or_a(state.rf.hl.l);
            break;
        case 0xB6:
            or_a(gb->mmu.read(state.rf.hl.v));
            break;
        case 0xB7:
            or_a(state.rf.af.a);
            break;
        case 0xB8:
            cp(state.rf.bc.b);
            break;
        case 0xB9:
            cp(state.rf.bc.c);
            break;
        case 0xBA:
            cp(state.rf.de.d);
            break;
        case 0xBB:
            cp(state.rf.de.e);
            break;
        case 0xBC:
            cp(state.rf.hl.h);
            break;
        case 0xBD:
            cp(state.rf.hl.l);
            break;
        case 0xBE:
            cp(gb->mmu.read(state.rf.hl.v));
            break;
        case 0xBF:
            cp(state.rf.af.a);
            break;
        case 0xC0:
            if (!is_set(state.rf.af.f, Flags::eZero))
            {
                state.branched = true;
                state.pc = stack_pop();
            }
            break;
        case 0xC1:
            state.rf.bc.v = stack_pop();
            break;
        case 0xC2:
            {
                const uint16_t addr = read_word_at_pc();
                if (!is_set(state.rf.af.f, Flags::eZero))
                {
                    state.branched = true;
                    state.pc = addr;
                }
            }
            break;
        case 0xC3:
            state.pc = read_word_at_pc();
            break;
        case 0xC4:
            {
                const uint16_t addr = read_word_at_pc();
                if (!is_set(state.rf.af.f, Flags::eZero))
                {
                    state.branched = true;
                    stack_push(state.pc);
                    state.pc = addr;
                }
            }
            break;
        case 0xC5:
            stack_push(state.rf.bc.v);
            break;
        case 0xC6:
            add_a(gb->mmu.read(state.pc++));
            break;
        case 0xC7:
            stack_push(state.pc);
            state.pc = 0x00;
            break;
        case 0xC8:
            if (is_set(state.rf.af.f, Flags::eZero))
            {
                state.branched = true;
                state.pc = stack_pop();
            }
            break;
        case 0xC9:
            state.pc = stack_pop();
            break;
        case 0xCA:
            {
                const uint16_t addr = read_word_at_pc();
                if (is_set(state.rf.af.f, Flags::eZero))
                {
                    state.branched = true;
                    state.pc = addr;
                }
            }
            break;
        case 0xCC:
            {
                const uint16_t addr = read_word_at_pc();
                if (is_set(state.rf.af.f, Flags::eZero))
                {
                    state.branched = true;
                    stack_push(state.pc);
                    state.pc = addr;
                }
            }
            break;
        case 0xCD:
            {
                const uint16_t addr = read_word_at_pc();
                stack_push(state.pc);
                state.pc = addr;
            }
            break;
        case 0xCE:
            adc_a(gb->mmu.read(state.pc++));
            break;
        case 0xCF:
            stack_push(state.pc);
            state.pc = 0x08;
            break;
        case 0xD0:
            if (!is_set(state.rf.af.f, Flags::eCarry))
            {
                state.branched = true;
                state.pc = stack_pop();
            }
            break;
        case 0xD1:
            state.rf.de.v = stack_pop();
            break;
        case 0xD2:
            {
                const uint16_t addr = read_word_at_pc();
                if (!is_set(state.rf.af.f, Flags::eCarry))
                {
                    state.branched = true;
                    state.pc = addr;
                }
            }
            break;
        case 0xD4:
            {
                const uint16_t addr = read_word_at_pc();
                if (!is_set(state.rf.af.f, Flags::eCarry))
                {
                    state.branched = true;
                    stack_push(state.pc);
                    state.pc = addr;
                }
            }
            break;
        case 0xD5:
            stack_push(state.rf.de.v);
            break;
        case 0xD6:
            sub_a(gb->mmu.read(state.pc++));
            break;
        case 0xD7:
            stack_push(state.pc);
            state.pc = 0x10;
            break;
        case 0xD8:
            if (is_set(state.rf.af.f, Flags::eCarry))
            {
                state.branched = true;
                state.pc = stack_pop();
            }
            break;
        case 0xD9:
            state.pc = stack_pop();
            state.interrupts_enabled = true;
            break;
        case 0xDA:
            {
                const uint16_t addr = read_word_at_pc();
                if (is_set(state.rf.af.f, Flags::eCarry))
                {
                    state.branched = true;
                    state.pc = addr;
                }
            }
            break;
        case 0xDC:
            {
                const uint16_t addr = read_word_at_pc();
                if (is_set(state.rf.af.f, Flags::eCarry))
                {
                    state.branched = true;
                    stack_push(state.pc);
                    state.pc = addr;
                }
            }
            break;
        case 0xDE:
            sbc_a(gb->mmu.read(state.pc++));
            break;
        case 0xDF:
            stack_push(state.pc);
            state.pc = 0x18;
            break;
        case 0xE0:
            gb->mmu.write(0xFF00 + gb->mmu.read(state.pc++), state.rf.af.a);
            break;
        case 0xE1:
            state.rf.hl.v = stack_pop();
            break;
        case 0xE2:
            gb->mmu.write(0xFF00 + state.rf.bc.c, state.rf.af.a);
            break;
        case 0xE5:
            stack_push(state.rf.hl.v);
            break;
        case 0xE6:
            and_a(gb->mmu.read(state.pc++));
            break;
        case 0xE7:
            stack_push(state.pc);
            state.pc = 0x20;
            break;
        case 0xE8:
            {
                const uint16_t sp = state.sp;
                const int8_t val = static_cast<int8_t>(gb->mmu.read(state.pc++));
                const int result = sp + val;
                state.sp = static_cast<uint16_t>(result);
                set_bit(state.rf.af.f, Flags::eZero, false);
                set_bit(state.rf.af.f, Flags::eSubtract, false);
                set_bit(state.rf.af.f, Flags::eCarry, ((sp ^ val ^ (result & 0xFFFF)) & 0x100) == 0x100);
                set_bit(state.rf.af.f, Flags::eHalfCarry, ((sp ^ val ^ (result & 0xFFFF)) & 0x10) == 0x10);
            }
            break;
        case 0xE9:
            state.pc = state.rf.hl.v;
            break;
        case 0xEA:
            gb->mmu.write(read_word_at_pc(), state.rf.af.a);
            break;
        case 0xEE:
            xor_a(gb->mmu.read(state.pc++));
            break;
        case 0xEF:
            stack_push(state.pc);
            state.pc = 0x28;
            break;
        case 0xF0:
            state.rf.af.a = gb->mmu.read(0xFF00 + gb->mmu.read(state.pc++));
            break;
        case 0xF1:
            state.rf.af.v = stack_pop();
            state.rf.af.f &= 0xF0;
            break;
        case 0xF2:
            state.rf.af.a = gb->mmu.read(0xFF00 + state.rf.bc.c);
            break;
        case 0xF3:
            state.interrupts_enabled = false;
            break;
        case 0xF5:
            stack_push(state.rf.af.v);
            break;
        case 0xF6:
            or_a(gb->mmu.read(state.pc++));
            break;
        case 0xF7:
            stack_push(state.pc);
            state.pc = 0x30;
            break;
        case 0xF8:
            {
                int8_t val = static_cast<int8_t>(gb->mmu.read(state.pc));
                const int result = static_cast<int>(state.sp + val);
                state.rf.hl.v = static_cast<uint16_t>(result);
                set_bit(state.rf.af.f, Flags::eZero, false);
                set_bit(state.rf.af.f, Flags::eSubtract, false);
                set_bit(state.rf.af.f, Flags::eCarry, ((state.sp ^ val ^ (result & 0xFFFF)) & 0x100) == 0x100);
                set_bit(state.rf.af.f, Flags::eHalfCarry, ((state.sp ^ val ^ (result & 0xFFFF)) & 0x10) == 0x10);
            }
            break;
        case 0xF9:
            state.sp = state.rf.hl.v;
            break;
        case 0xFA:
            state.rf.af.a = gb->mmu.read(read_word_at_pc());
            break;
        case 0xFB:
            state.interrupts_enabled = true;
            break;
        case 0xFE:
            cp(gb->mmu.read(state.pc++));
            break;
        case 0xFF:
            stack_push(state.pc);
            state.pc = 0x38;
            break;
        default:
            break;
        }
    }
    else
    {
        next_inst = gb->mmu.read(state.pc++);
        const auto rindex = (next_inst & 0b111) ^ 1;
        switch (next_inst)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03:
        case 0x04:
        case 0x05:
        case 0x07:
            state.rf.r[rindex] = rlc(state.rf.r[rindex]);
            break;
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B:
        case 0x0C:
        case 0x0D:
        case 0x0F:
            state.rf.r[rindex] = rrc(state.rf.r[rindex]);
            break;
        case 0x10:
        case 0x11:
        case 0x12:
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x17:
            state.rf.r[rindex] = rl(state.rf.r[rindex]);
            break;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B:
        case 0x1C:
        case 0x1D:
        case 0x1F:
            state.rf.r[rindex] = rr(state.rf.r[rindex]);
            break;
        case 0x20:
        case 0x21:
        case 0x22:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x27:
            state.rf.r[rindex] = sla(state.rf.r[rindex]);
            break;
        case 0x28:
        case 0x29:
        case 0x2A:
        case 0x2B:
        case 0x2C:
        case 0x2D:
        case 0x2F:
            state.rf.r[rindex] = sra(state.rf.r[rindex]);
            break;
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x35:
        case 0x37:
            state.rf.r[rindex] = swap(state.rf.r[rindex]);
            break;
        case 0x38:
        case 0x39:
        case 0x3A:
        case 0x3B:
        case 0x3C:
        case 0x3D:
        case 0x3F:
            state.rf.r[rindex] = srl(state.rf.r[rindex]);
            break;
        case 0x40:
        case 0x48:
        case 0x50:
        case 0x58:
        case 0x60:
        case 0x68:
        case 0x70:
        case 0x78:
        case 0x41:
        case 0x49:
        case 0x51:
        case 0x59:
        case 0x61:
        case 0x69:
        case 0x71:
        case 0x79:
        case 0x42:
        case 0x4A:
        case 0x52:
        case 0x5A:
        case 0x62:
        case 0x6A:
        case 0x72:
        case 0x7A:
        case 0x43:
        case 0x4B:
        case 0x53:
        case 0x5B:
        case 0x63:
        case 0x6B:
        case 0x73:
        case 0x7B:
        case 0x44:
        case 0x4C:
        case 0x54:
        case 0x5C:
        case 0x64:
        case 0x6C:
        case 0x74:
        case 0x7C:
        case 0x45:
        case 0x4D:
        case 0x55:
        case 0x5D:
        case 0x65:
        case 0x6D:
        case 0x75:
        case 0x7D:
        case 0x47:
        case 0x4F:
        case 0x57:
        case 0x5F:
        case 0x67:
        case 0x6F:
        case 0x77:
        case 0x7F:
            bit(state.rf.r[rindex], (next_inst >> 3) - 8);
            break;
        case 0x80:
        case 0x88:
        case 0x90:
        case 0x98:
        case 0xA0:
        case 0xA8:
        case 0xB0:
        case 0xB8:
        case 0x81:
        case 0x89:
        case 0x91:
        case 0x99:
        case 0xA1:
        case 0xA9:
        case 0xB1:
        case 0xB9:
        case 0x82:
        case 0x8A:
        case 0x92:
        case 0x9A:
        case 0xA2:
        case 0xAA:
        case 0xB2:
        case 0xBA:
        case 0x83:
        case 0x8B:
        case 0x93:
        case 0x9B:
        case 0xA3:
        case 0xAB:
        case 0xB3:
        case 0xBB:
        case 0x84:
        case 0x8C:
        case 0x94:
        case 0x9C:
        case 0xA4:
        case 0xAC:
        case 0xB4:
        case 0xBC:
        case 0x85:
        case 0x8D:
        case 0x95:
        case 0x9D:
        case 0xA5:
        case 0xAD:
        case 0xB5:
        case 0xBD:
        case 0x87:
        case 0x8F:
        case 0x97:
        case 0x9F:
        case 0xA7:
        case 0xAF:
        case 0xB7:
        case 0xBF:
            state.rf.r[rindex] &= ~(1 << (((next_inst - 0x40) >> 3) - 8));
            break;
        case 0xC0:
        case 0xC8:
        case 0xD0:
        case 0xD8:
        case 0xE0:
        case 0xE8:
        case 0xF0:
        case 0xF8:
        case 0xC1:
        case 0xC9:
        case 0xD1:
        case 0xD9:
        case 0xE1:
        case 0xE9:
        case 0xF1:
        case 0xF9:
        case 0xC2:
        case 0xCA:
        case 0xD2:
        case 0xDA:
        case 0xE2:
        case 0xEA:
        case 0xF2:
        case 0xFA:
        case 0xC3:
        case 0xCB:
        case 0xD3:
        case 0xDB:
        case 0xE3:
        case 0xEB:
        case 0xF3:
        case 0xFB:
        case 0xC4:
        case 0xCC:
        case 0xD4:
        case 0xDC:
        case 0xE4:
        case 0xEC:
        case 0xF4:
        case 0xFC:
        case 0xC5:
        case 0xCD:
        case 0xD5:
        case 0xDD:
        case 0xE5:
        case 0xED:
        case 0xF5:
        case 0xFD:
        case 0xC7:
        case 0xCF:
        case 0xD7:
        case 0xDF:
        case 0xE7:
        case 0xEF:
        case 0xF7:
        case 0xFF:
            state.rf.r[rindex] |= 1 << (((next_inst - 0x80) >> 3) - 8);
            break;
        case 0x1E:
            gb->mmu.write(state.rf.hl.v, rr(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x26:
            gb->mmu.write(state.rf.hl.v, sla(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x2E:
            gb->mmu.write(state.rf.hl.v, sra(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x06:
            gb->mmu.write(state.rf.hl.v, rlc(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x0E:
            gb->mmu.write(state.rf.hl.v, rrc(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x16:
            gb->mmu.write(state.rf.hl.v, rl(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x36:
            gb->mmu.write(state.rf.hl.v, swap(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x3E:
            gb->mmu.write(state.rf.hl.v, srl(gb->mmu.read(state.rf.hl.v)));
            break;
        case 0x46:
        case 0x4E:
        case 0x56:
        case 0x5E:
        case 0x66:
        case 0x6E:
        case 0x76:
        case 0x7E:
            bit(gb->mmu.read(state.rf.hl.v), (next_inst >> 3) - 8);
            break;
        case 0x86:
        case 0x8E:
        case 0x96:
        case 0x9E:
        case 0xA6:
        case 0xAE:
        case 0xB6:
        case 0xBE:
            gb->mmu.write(state.rf.hl.v, gb->mmu.read(state.rf.hl.v) & ~(1 << (((next_inst - 0x40) >> 3) - 8)));
            break;
        case 0xC6:
        case 0xCE:
        case 0xD6:
        case 0xDE:
        case 0xE6:
        case 0xEE:
        case 0xF6:
        case 0xFE:
            gb->mmu.write(state.rf.hl.v, gb->mmu.read(state.rf.hl.v) | (1 << (((next_inst - 0x80) >> 3) - 8)));
            break;
        default:
            break;
        }
    }

    uint8_t cycles{};
    if (inst != 0xCB)
    {
        if (state.branched)
            cycles = CYCLES_BRANCHED[inst];
        else
            cycles = CYCLES_NORMAL[inst];
    }
    else
        cycles = CYCLES_2B[next_inst];
    cycles_emulated += cycles;
    return cycles;
}

__device__ void PPUCuda::tick(uint64_t cycles)
{
    cycle_counter += cycles;

    switch (mode)
    {
    case VideoMode::eAccessOAM:
        if (cycle_counter >= CLOCKS_PER_SCANLINE_OAM)
        {
            cycle_counter = cycle_counter % CLOCKS_PER_SCANLINE_OAM;
            lcd_status |= 0x11;
            mode = VideoMode::eAccessVRAM;
        }
        break;
    case VideoMode::eAccessVRAM:
        if (cycle_counter >= CLOCKS_PER_SCANLINE_VRAM)
        {
            cycle_counter = cycle_counter % CLOCKS_PER_SCANLINE_VRAM;
            mode = VideoMode::eHBlank;

            if (lcd_status & (1 << 3))
                gb->cpu.set_interrupt_flag(1, true);

            const bool ly_coincidence = ly_cmp == line;
            if ((lcd_status & (1 << 6)) && ly_coincidence)
                gb->cpu.set_interrupt_flag(1, true);

            if (ly_coincidence)
                lcd_status |= 0b00000100;
            else
                lcd_status &= 0b11111011;
            lcd_status &= 0b11111100;
        }
        break;
    case VideoMode::eHBlank:
        if (cycle_counter >= CLOCKS_PER_HBLANK)
        {
            cycle_counter = cycle_counter % CLOCKS_PER_HBLANK;

            if (control_byte & (1 << 7))
            {
                // generate a line of pixels (if the display is enabled)
                if (control_byte & 1)
                {
                    // draw background if it's enabled
                    const bool use_tile_set_zero = control_byte & (1 << 4);
                    const bool use_tile_map_zero = !(control_byte & (1 << 3));

                    const uint8_t palette[4] = {
                        uint8_t(bg_palette & 0b00000011),
                        uint8_t((bg_palette & 0b00001100) >> 2),
                        uint8_t((bg_palette & 0b00110000) >> 4),
                        uint8_t((bg_palette & 0b11000000) >> 6)
                    };

                    const uint16_t tile_set_address = use_tile_set_zero ? TILE_SET_ZERO_ADDRESS : TILE_SET_ONE_ADDRESS;
                    const uint16_t tile_map_address = use_tile_map_zero ? TILE_MAP_ZERO_ADDRESS : TILE_MAP_ONE_ADDRESS;

                    for (uint32_t x = 0; x < FB_WIDTH; x++)
                    {
                        const uint32_t scrolled_x = x + scroll_x;
                        const uint32_t scrolled_y = line + scroll_y;
                        const uint32_t bg_map_x = scrolled_x % BG_MAP_SIZE;
                        const uint32_t by_map_y = scrolled_y % BG_MAP_SIZE;
                        const uint32_t tile_x = bg_map_x / TILE_WIDTH_PX;
                        const uint32_t tile_y = by_map_y / TILE_HEIGHT_PX;
                        const uint32_t tile_pixel_x = bg_map_x % TILE_WIDTH_PX;
                        const uint32_t tile_pixel_y = by_map_y % TILE_HEIGHT_PX;
                        const uint32_t tile_index = tile_y * TILES_PER_LINE + tile_x;
                        const uint16_t tile_id_address = tile_map_address + tile_index;
                        const uint8_t tile_id = gb->mmu.read(tile_id_address);
                        const uint16_t tile_data_mem_offset = use_tile_set_zero ? (tile_id * TILE_BYTES) : (static_cast<int8_t>(tile_id) + 128) * TILE_BYTES;
                        const uint16_t tile_data_line_offset = tile_pixel_y * 2;
                        const uint16_t tile_line_data_start_address = tile_set_address + tile_data_mem_offset + tile_data_line_offset;
                        const uint8_t pixels_1 = gb->mmu.read(tile_line_data_start_address);
                        const uint8_t pixels_2 = gb->mmu.read(tile_line_data_start_address + 1);

                        const uint8_t color_u8 = static_cast<uint8_t>((((pixels_2 >> (7 - tile_pixel_x)) & 1) << 1) | ((pixels_1 >> (7 - tile_pixel_x)) & 1));
                        set_pixel(x, line, get_gray(get_color(palette[color_u8])));
                    }
                }
                if (control_byte & (1 << 5))
                {
                    // draw windows if it's enabled
                    const uint32_t scrolled_y = line - window_y;
                    if (scrolled_y < FB_HEIGHT)
                    {
                        const bool use_tile_set_zero = control_byte & (1 << 4);
                        const bool use_tile_map_zero = !(control_byte & (1 << 6));

                        const uint8_t palette[4] = {
                            uint8_t(bg_palette & 0b00000011),
                            uint8_t((bg_palette & 0b00001100) >> 2),
                            uint8_t((bg_palette & 0b00110000) >> 4),
                            uint8_t((bg_palette & 0b11000000) >> 6)
                        };

                        const uint16_t tile_set_address = use_tile_set_zero ? TILE_SET_ZERO_ADDRESS : TILE_SET_ONE_ADDRESS;
                        const uint16_t tile_map_address = use_tile_map_zero ? TILE_MAP_ZERO_ADDRESS : TILE_MAP_ONE_ADDRESS;

                        for (uint32_t x = 0; x < FB_WIDTH; x++)
                        {
                            const uint16_t scrolled_x = x + window_x - 7;
                            const uint16_t tile_x = scrolled_x / TILE_WIDTH_PX;
                            const uint16_t tile_y = scrolled_y / TILE_HEIGHT_PX;
                            const uint16_t tile_pixel_x = scrolled_x % TILE_WIDTH_PX;
                            const uint16_t tile_pixel_y = scrolled_y % TILE_HEIGHT_PX;
                            const uint16_t tile_index = tile_y * TILES_PER_LINE + tile_x;
                            const uint16_t tile_id_address = tile_map_address + tile_index;
                            const uint8_t tile_id = gb->mmu.read(tile_id_address);
                            const uint16_t tile_data_mem_offset = use_tile_set_zero ? tile_id * TILE_BYTES : (static_cast<int8_t>(tile_id) + 128) * TILE_BYTES;
                            const uint16_t tile_data_line_offset = tile_pixel_y * 2;
                            const uint16_t tile_line_data_start_address = tile_set_address + tile_data_mem_offset + tile_data_line_offset;
                            const uint8_t pixels_1 = gb->mmu.read(tile_line_data_start_address);
                            const uint8_t pixels_2 = gb->mmu.read(tile_line_data_start_address + 1);
                            const uint8_t color_u8 = static_cast<uint8_t>((((pixels_2 >> (7 - tile_pixel_x)) & 1) << 1) | ((pixels_1 >> (7 - tile_pixel_x)) & 1));
                            set_pixel(x, line, get_gray(get_color(palette[color_u8])));
                        }
                    }
                }
            }

            line++;

            if (line == 144)
            {
                mode = VideoMode::eVBlank;
                gb->cpu.set_interrupt_flag(0, true);
                lcd_status &= 0b11111101;
                lcd_status |= 0b00000001;
            }
            else
            {
                lcd_status &= 0b11111110;
                lcd_status |= 0b00000010;
                mode = VideoMode::eAccessOAM;
            }
        }
        break;
    case VideoMode::eVBlank:
        if (cycle_counter >= CLOCKS_PER_SCANLINE)
        {
            cycle_counter = cycle_counter % CLOCKS_PER_SCANLINE;
            line++;
            if (line == 154) {
                if (control_byte & (1 << 1))
                {
                    // draw sprites here, if enabled
                    for (uint16_t sprite_no = 0; sprite_no < 40; ++sprite_no)
                    {
                        const uint16_t offset_in_oam = sprite_no * SPRITE_BYTES;
                        const uint16_t oam_start = 0xFE00 + offset_in_oam;
                        const uint8_t sprite_y = gb->mmu.read(oam_start);
                        const uint8_t sprite_x = gb->mmu.read(oam_start + 1);

                        // only draw visible sprites
                        if (!((sprite_y == 0 || sprite_y >= 160) || (sprite_x == 0 || sprite_x >= 168)))
                        {
                            const uint16_t sprite_size_multiplier = (control_byte & (1 << 2)) ? 2 : 1;
                            const uint16_t tile_set_location = TILE_SET_ZERO_ADDRESS;

                            const uint8_t pattern_n = gb->mmu.read(oam_start + 2);
                            const uint8_t sprite_attrs = gb->mmu.read(oam_start + 3);

                            const bool use_palette_1 = sprite_attrs & (1 << 4);
                            const bool flip_x = sprite_attrs & (1 << 5);
                            const bool flip_y = sprite_attrs & (1 << 6);
                            const bool obj_behind_bg = sprite_attrs & (1 << 7);

                            const uint8_t sprite_palette = use_palette_1 ? sprite_palette_1 : sprite_palette_0;
                            const uint8_t palette[4] = {
                                uint8_t(sprite_palette & 0b00000011),
                                uint8_t((sprite_palette & 0b00001100) >> 2),
                                uint8_t((sprite_palette & 0b00110000) >> 4),
                                uint8_t((sprite_palette & 0b11000000) >> 6)
                            };

                            const uint16_t tile_offset = pattern_n * TILE_BYTES;
                            const uint16_t pattern_address = tile_set_location + tile_offset;

                            int start_y = sprite_y - 16;
                            int start_x = sprite_x - 8;
                            uint8_t tile_data[2 * TILE_HEIGHT_PX];
                            for (uint16_t i = 0; i < TILE_HEIGHT_PX * sprite_size_multiplier; ++i)
                                tile_data[i] = gb->mmu.read(pattern_address + 2 * i);

                            for (uint16_t y = 0; y < TILE_HEIGHT_PX * sprite_size_multiplier; y++)
                            {
                                for (uint16_t x = 0; x < TILE_WIDTH_PX; x++)
                                {
                                    const uint16_t tile_y = !flip_y ? y : (TILE_HEIGHT_PX * sprite_size_multiplier) - y - 1;
                                    const uint16_t tile_x = !flip_x ? x : TILE_WIDTH_PX - x - 1;
                                    const uint8_t offset_to_byte = tile_y * 2;
                                    const uint8_t color_u8 = static_cast<uint8_t>((((tile_data[offset_to_byte + 1] >> (7 - tile_x)) & 1) << 1) |
                                        ((tile_data[offset_to_byte] >> (7 - tile_x)) & 1));

                                    if (!color_u8)
                                        continue;

                                    const int screen_x = start_x + x;
                                    const int screen_y = start_y + y;
                                    if (screen_x >= FB_WIDTH || screen_y >= FB_HEIGHT)
                                        continue;
                                    if (obj_behind_bg && get_pixel(screen_x, screen_y) != 0xFF)
                                        continue;

                                    set_pixel(screen_x, screen_y, get_gray(get_color(palette[color_u8])));
                                }
                            }
                        }
                    }
                }
                line = 0;
                mode = VideoMode::eAccessOAM;
                lcd_status &= 0b11111110;
                lcd_status |= 0b00000010;
            };
        }
        break;
    }
}

__device__ void CPUCuda::set_interrupt_flag(uint8_t no, bool value)
{
    set_bit(state.interrupt_flag, no, value);
}

__device__ void CPUCuda::set_interrupt_enabled(uint8_t value)
{
    state.interrupt_enabled = value;
}

__device__ uint8_t CPUCuda::get_interrupt_enabled() const
{
    return state.interrupt_enabled;
}

__device__ uint8_t CPUCuda::get_interrupt_flag() const
{
    return state.interrupt_flag;
}

__device__ void GameboyCuda::tick(uint16_t ms)
{
    const uint64_t CLOCK_RATE = 4194304;
    const auto cycles_to_emulate = CLOCK_RATE * (ms / 1000.0f);
    //mmu.startup = true;
    uint64_t cycles_emulated{};
    while (cycles_emulated < cycles_to_emulate)
    {
        // step CPU, get # of cycles
        const auto cycles = cpu.tick();
        // step PPU
        ppu.tick(cycles);
        cycles_emulated += cycles;
    }
}

__global__ void emulateKernel(GameboyCuda* gb)
{
    GameboyCuda* gb_specific = gb + threadIdx.x + blockIdx.x * EMU_WIDTH;
    gb_specific->cpu.gb = gb_specific;
    gb_specific->mmu.gb = gb_specific;
    gb_specific->ppu.gb = gb_specific;
    gb_specific->ppu.display = reinterpret_cast<uint8_t*>(gb + EMU_WIDTH * EMU_HEIGHT);
    gb_specific->tick(16);
    gb_specific->ppu.gb = gb_specific;
}

GameboyCuda* gb_data = nullptr;

void initGpuData(const uint8_t** roms, uint64_t roms_size)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&gb_data, EMU_HEIGHT * EMU_WIDTH * (sizeof(GameboyCuda) + FB_HEIGHT * FB_WIDTH * 3));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!");
    GameboyCuda* g = new GameboyCuda[EMU_HEIGHT * EMU_WIDTH];
    memset(g, 0, EMU_HEIGHT * EMU_WIDTH * sizeof(GameboyCuda));
    for (uint64_t i = 0; i < EMU_HEIGHT * EMU_WIDTH; ++i)
    {
        g[i].mmu.startup = true;
        memcpy(g[i].cartridge.rom, roms[(i + i / EMU_WIDTH) % roms_size], 32 * 1024);
    }

    cudaMemcpy(gb_data, g, EMU_HEIGHT * EMU_WIDTH * sizeof(GameboyCuda), cudaMemcpyHostToDevice);
    delete[] g;
}

cudaError_t emulateGpu(uint8_t* display)
{
    emulateKernel<<<EMU_HEIGHT, EMU_WIDTH>>>(gb_data);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "emulateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching emulateKernel!\n", cudaStatus);

    const uint8_t* offset = reinterpret_cast<uint8_t*>(gb_data) + EMU_HEIGHT * EMU_WIDTH * sizeof(GameboyCuda);
    cudaMemcpy(display, offset, EMU_HEIGHT * EMU_WIDTH * FB_HEIGHT * FB_WIDTH * 3, cudaMemcpyDeviceToHost);

    return cudaStatus;
}
