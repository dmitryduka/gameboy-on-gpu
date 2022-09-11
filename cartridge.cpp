#include "cartridge.h"
#include <fstream>
#include <array>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace
{
    namespace header {
        constexpr int entry_point = 0x100;
        constexpr int logo = 0x104;
        constexpr int title = 0x134;
        constexpr int manufacturer_code = 0x13F;
        constexpr int cgb_flag = 0x143;
        constexpr int new_license_code = 0x144;
        constexpr int sgb_flag = 0x146;
        constexpr int cartridge_type = 0x147;
        constexpr int rom_size = 0x148;
        constexpr int ram_size = 0x149;
        constexpr int destination_code = 0x14A;
        constexpr int old_license_code = 0x14B;
        constexpr int version_number = 0x14C;
        constexpr int header_checksum = 0x14D;
        constexpr int global_checksum = 0x14E;
    }

    std::unordered_map<uint8_t, CartridgeType> cartidge_type {
        {0x00, CartridgeType::eROMOnly},
        {0x08, CartridgeType::eROMOnly},
        {0x09, CartridgeType::eROMOnly},
        {0x01, CartridgeType::eMBC1},
        {0x02, CartridgeType::eMBC1},
        {0x03, CartridgeType::eMBC1},
        {0xff, CartridgeType::eMBC1},
        {0x05, CartridgeType::eMBC2},
        {0x06, CartridgeType::eMBC2},
        {0x0F, CartridgeType::eMBC3},
        {0x10, CartridgeType::eMBC3},
        {0x11, CartridgeType::eMBC3},
        {0x12, CartridgeType::eMBC3},
        {0x13, CartridgeType::eMBC3},
        {0x15, CartridgeType::eMBC4},
        {0x16, CartridgeType::eMBC4},
        {0x17, CartridgeType::eMBC4},
        {0x19, CartridgeType::eMBC5},
        {0x1A, CartridgeType::eMBC5},
        {0x1B, CartridgeType::eMBC5},
        {0x1C, CartridgeType::eMBC5},
        {0x1D, CartridgeType::eMBC5},
        {0x1E, CartridgeType::eMBC5}
    };

    std::unordered_map<uint8_t, ROMSize> rom_sizes{
        {0x00, ROMSize::eKB32},
        {0x01, ROMSize::eKB64},
        {0x02, ROMSize::eKB128},
        {0x03, ROMSize::eKB256},
        {0x04, ROMSize::eKB512},
        {0x05, ROMSize::eMB1},
        {0x06, ROMSize::eMB2},
        {0x07, ROMSize::eMB4},
        {0x52, ROMSize::eMB1p1},
        {0x53, ROMSize::eMB1p2},
        {0x54, ROMSize::eMB1p5}
    };

    std::unordered_map<uint8_t, RAMSize> ram_sizes{
        {0x00, RAMSize::eNone},
        {0x01, RAMSize::eKB2},
        {0x02, RAMSize::eKB8},
        {0x03, RAMSize::eKB32},
        {0x04, RAMSize::eKB128},
        {0x05, RAMSize::eKB64}
    };

    std::string to_string(CartridgeType ctype)
    {
        switch (ctype)
        {
        case CartridgeType::eROMOnly:
            return "ROM only";
        case CartridgeType::eMBC1:
            return "MBC1";
        case CartridgeType::eMBC2:
            return "MBC2";
        case CartridgeType::eMBC3:
            return "MBC3";
        case CartridgeType::eMBC4:
            return "MBC4";
        case CartridgeType::eMBC5:
            return "MBC5";
        default:
            return "unknown";
        }
    }

    std::string to_string(ROMSize size)
    {
        switch (size)
        {
        case ROMSize::eKB32:
            return "32KB (no ROM banking)";
        case ROMSize::eKB64:
            return "64KB (4 banks)";
        case ROMSize::eKB128:
            return "128KB (8 banks)";
        case ROMSize::eKB256:
            return "256KB (16 banks)";
        case ROMSize::eKB512:
            return "512KB (32 banks)";
        case ROMSize::eMB1:
            return "1MB (64 banks)";
        case ROMSize::eMB2:
            return "2MB (128 banks)";
        case ROMSize::eMB4:
            return "4MB (256 banks)";
        case ROMSize::eMB1p1:
            return "1.1MB (72 banks)";
        case ROMSize::eMB1p2:
            return "1.2MB (80 banks)";
        case ROMSize::eMB1p5:
            return "1.5MB (96 banks)";
        default:
            return "unknown";
        }
    }

    std::string to_string(RAMSize size)
    {
        switch (size)
        {
        case RAMSize::eNone:
            return "No RAM";
        case RAMSize::eKB2:
            return "2KB";
        case RAMSize::eKB8:
            return "8KB";
        case RAMSize::eKB32:
            return "32KB";
        case RAMSize::eKB128:
            return "128KB";
        case RAMSize::eKB64:
            return "64KB";
        default:
            return "unknown";
        }
    }

    std::vector<uint8_t> load_file(const std::filesystem::path& rom_path)
    {
        std::ifstream in(rom_path, std::ios::in | std::ios::binary);

        if (in.good())
        {
            std::istreambuf_iterator<char> start(in), end;
            return std::vector<uint8_t>(start, end);
        }
        spdlog::error("Error reading the file {}", rom_path.string());
        return {};
    }
}


CartridgeInfo::CartridgeInfo(const std::vector<uint8_t>& rom)
{
    if (rom.empty())
        return;

    version = rom[header::version_number];
    type = cartidge_type[rom[header::cartridge_type]];
    rom_size = rom_sizes[rom[header::rom_size]];
    ram_size = ram_sizes[rom[header::ram_size]];

    const int TITLE_LENGTH = 11;
    title.assign(reinterpret_cast<const char*>(&rom[header::title]), TITLE_LENGTH);
}

Cartridge::Cartridge(const char* rom_path) : rom(load_file(rom_path)), info(rom)
{
    spdlog::info("Cartidge loaded from '{}'", rom_path);
    spdlog::info("------------------------------------------------");
    spdlog::info("Title: '{}' (version {})", info.title, info.version);
    spdlog::info("Cartridge type: {}", to_string(info.type));
    spdlog::info("ROM/RAM: {} / {}", to_string(info.rom_size), to_string(info.ram_size));
    spdlog::info("------------------------------------------------");
}

uint8_t Cartridge::read(const uint16_t& address) const
{
    return rom[address];
}

const std::vector<uint8_t>& Cartridge::get_rom() const
{
    return rom;
}